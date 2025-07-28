/*
 * ESP32-S3 Velocity-Sensitive Keyboard Scanner
 * Scans keyboard matrix with velocity detection via dual switches per key
 * Sends MIDI data to Raspberry Pi via USB Serial
 */

#include <Arduino.h>

// Timing constants for velocity detection
#define VELOCITY_TIMEOUT_US 10000  // 10ms max time between MK and BR
#define SCAN_INTERVAL_US 1000      // 1ms scan interval (1kHz)
#define DEBOUNCE_TIME_US 2000      // 2ms debounce time

// Row pins (T0-T7) - outputs
const int ROW_PINS[] = {15, 16, 17, 18, 8, 3, 46, 9};
const int NUM_ROWS = 8;

// Bass section pins
const int BASS_MK_PINS[] = {10, 11, 12, 13, 14};  // MK0-MK4
const int BASS_BR_PINS[] = {38, 37, 36, 35, 45};  // BR0-BR4
const int NUM_BASS_KEYS = 5;

// Treble section pins  
const int TREBLE_MK_PINS[] = {1, 2, 42, 41, 39, 40};      // MK5-MK10
const int TREBLE_BR_PINS[] = {48, 47, 4, 5, 6, 7};        // BR5-BR10
const int NUM_TREBLE_KEYS = 6;

// Pedal pins
const int PEDAL_PINS[] = {21, 20, 19};  // Left, Center, Right
const int NUM_PEDALS = 3;

// Key state tracking
struct KeyState {
  bool mk_pressed;        // Make switch state
  bool br_pressed;        // Break switch state
  bool key_active;        // Overall key state
  uint32_t mk_time;       // Time when MK switch activated (microseconds)
  uint32_t br_time;       // Time when BR switch activated  
  uint32_t last_change;   // Last state change time (for debouncing)
  uint8_t velocity;       // Calculated velocity (0-127)
  bool velocity_sent;     // Whether velocity has been sent for this press
};

// Key state arrays
KeyState bass_keys[NUM_BASS_KEYS][NUM_ROWS];
KeyState treble_keys[NUM_TREBLE_KEYS][NUM_ROWS]; 
KeyState pedal_states[NUM_PEDALS];

// MIDI note mapping - adjust these to match your keyboard layout
const int BASS_MIDI_START = 36;    // C2
const int TREBLE_MIDI_START = 60;  // C4 (Middle C)
const int PEDAL_MIDI_NOTES[] = {64, 67, 71};  // Sustain pedal CC numbers

void setup() {
  Serial.begin(115200);
  
  // Setup row pins as outputs (initially HIGH)
  for (int i = 0; i < NUM_ROWS; i++) {
    pinMode(ROW_PINS[i], OUTPUT);
    digitalWrite(ROW_PINS[i], HIGH);
  }
  
  // Setup bass MK/BR pins as inputs with pullups
  for (int i = 0; i < NUM_BASS_KEYS; i++) {
    pinMode(BASS_MK_PINS[i], INPUT_PULLUP);
    pinMode(BASS_BR_PINS[i], INPUT_PULLUP);
  }
  
  // Setup treble MK/BR pins as inputs with pullups  
  for (int i = 0; i < NUM_TREBLE_KEYS; i++) {
    pinMode(TREBLE_MK_PINS[i], INPUT_PULLUP);
    pinMode(TREBLE_BR_PINS[i], INPUT_PULLUP);
  }
  
  // Setup pedal pins as inputs with pullups
  for (int i = 0; i < NUM_PEDALS; i++) {
    pinMode(PEDAL_PINS[i], INPUT_PULLUP);
  }
  
  // Initialise key states
  InitialiseKeyStates();
  
  Serial.println("ESP32-S3 Keyboard Scanner Ready");
  Serial.println("Format: NOTE_ON/OFF,note,velocity");
}

void InitialiseKeyStates() {
  // Initialise bass keys
  for (int key = 0; key < NUM_BASS_KEYS; key++) {
    for (int row = 0; row < NUM_ROWS; row++) {
      bass_keys[key][row] = {false, false, false, 0, 0, 0, 0, false};
    }
  }
  
  // Initialise treble keys
  for (int key = 0; key < NUM_TREBLE_KEYS; key++) {
    for (int row = 0; row < NUM_ROWS; row++) {
      treble_keys[key][row] = {false, false, false, 0, 0, 0, 0, false};
    }
  }
  
  // Initialise pedals
  for (int i = 0; i < NUM_PEDALS; i++) {
    pedal_states[i] = {false, false, false, 0, 0, 0, 0, false};
  }
}

uint8_t calculateVelocity(uint32_t mk_time, uint32_t br_time) {
  if (mk_time == 0 || br_time == 0 || br_time <= mk_time) {
    return 64; // Default medium velocity
  }
  
  uint32_t time_diff = br_time - mk_time;
  
  // Convert time difference to velocity (faster = higher velocity)
  // Time range: 100us (very fast) to 10000us (very slow)
  if (time_diff < 100) time_diff = 100;
  if (time_diff > VELOCITY_TIMEOUT_US) time_diff = VELOCITY_TIMEOUT_US;
  
  // Invert the relationship: shorter time = higher velocity
  uint8_t velocity = map(time_diff, 100, VELOCITY_TIMEOUT_US, 127, 1);
  return constrain(velocity, 1, 127);
}

void scanMatrix() {
  uint32_t current_time = micros();
  
  // Scan each row
  for (int row = 0; row < NUM_ROWS; row++) {
    // Set current row LOW, others HIGH
    for (int i = 0; i < NUM_ROWS; i++) {
      digitalWrite(ROW_PINS[i], (i == row) ? LOW : HIGH);
    }
    
    // Small delay for signal settling
    delayMicroseconds(10);
    
    // Scan bass keys for this row
    for (int key = 0; key < NUM_BASS_KEYS; key++) {
      scanKey(&bass_keys[key][row], 
              digitalRead(BASS_MK_PINS[key]) == LOW,
              digitalRead(BASS_BR_PINS[key]) == LOW,
              BASS_MIDI_START + (key * NUM_ROWS) + row,
              current_time);
    }
    
    // Scan treble keys for this row
    for (int key = 0; key < NUM_TREBLE_KEYS; key++) {
      scanKey(&treble_keys[key][row],
              digitalRead(TREBLE_MK_PINS[key]) == LOW,
              digitalRead(TREBLE_BR_PINS[key]) == LOW, 
              TREBLE_MIDI_START + (key * NUM_ROWS) + row,
              current_time);
    }
  }
  
  // Reset all rows HIGH
  for (int i = 0; i < NUM_ROWS; i++) {
    digitalWrite(ROW_PINS[i], HIGH);
  }
}

void scanKey(KeyState* key, bool mk_active, bool br_active, int midi_note, uint32_t current_time) {
  // Debouncing check
  if (current_time - key->last_change < DEBOUNCE_TIME_US) {
    return;
  }
  
  bool state_changed = false;
  
  // Check MK switch (first contact)
  if (mk_active && !key->mk_pressed) {
    key->mk_pressed = true;
    key->mk_time = current_time;
    key->last_change = current_time;
    state_changed = true;
  } else if (!mk_active && key->mk_pressed) {
    key->mk_pressed = false;
    key->last_change = current_time;
    state_changed = true;
  }
  
  // Check BR switch (second contact)  
  if (br_active && !key->br_pressed) {
    key->br_pressed = true;
    key->br_time = current_time;
    key->last_change = current_time;
    state_changed = true;
  } else if (!br_active && key->br_pressed) {
    key->br_pressed = false;
    key->last_change = current_time;
    state_changed = true;
  }
  
  // Determine key press state and calculate velocity
  bool new_key_active = key->mk_pressed || key->br_pressed;
  
  if (new_key_active && !key->key_active) {
    // Key press detected
    key->key_active = true;
    
    // Calculate velocity if we have both MK and BR times
    if (key->mk_time > 0 && key->br_time > 0) {
      key->velocity = calculateVelocity(key->mk_time, key->br_time);
    } else {
      key->velocity = 64; // Default velocity if only one switch detected
    }
    
    // Send note on
    Serial.print("NOTE_ON,");
    Serial.print(midi_note);
    Serial.print(",");
    Serial.println(key->velocity);
    
    key->velocity_sent = true;
    
  } else if (!new_key_active && key->key_active) {
    // Key release detected
    key->key_active = false;
    
    // Send note off
    Serial.print("NOTE_OFF,");
    Serial.print(midi_note);
    Serial.println(",0");
    
    // Reset timing data
    key->mk_time = 0;
    key->br_time = 0;
    key->velocity_sent = false;
  }
  
  // Handle velocity timeout (if MK activated but BR never comes)
  if (key->mk_pressed && !key->br_pressed && 
      key->mk_time > 0 && 
      (current_time - key->mk_time) > VELOCITY_TIMEOUT_US &&
      !key->velocity_sent && new_key_active) {
    
    // Send note with default velocity
    key->velocity = 64;
    Serial.print("NOTE_ON,");
    Serial.print(midi_note);
    Serial.print(",");
    Serial.println(key->velocity);
    key->velocity_sent = true;
  }
}

void scanPedals() {
  static uint32_t last_pedal_scan = 0;
  uint32_t current_time = micros();
  
  // Scan pedals at lower frequency (every 10ms)
  if (current_time - last_pedal_scan < 10000) {
    return;
  }
  last_pedal_scan = current_time;
  
  for (int i = 0; i < NUM_PEDALS; i++) {
    bool pedal_active = digitalRead(PEDAL_PINS[i]) == LOW;
    
    if (pedal_active && !pedal_states[i].key_active) {
      // Pedal pressed
      pedal_states[i].key_active = true;
      Serial.print("PEDAL_ON,");
      Serial.print(PEDAL_MIDI_NOTES[i]);
      Serial.println(",127");
      
    } else if (!pedal_active && pedal_states[i].key_active) {
      // Pedal released
      pedal_states[i].key_active = false;
      Serial.print("PEDAL_OFF,");
      Serial.print(PEDAL_MIDI_NOTES[i]);
      Serial.println(",0");
    }
  }
}

void loop() {
  static uint32_t last_scan = 0;
  uint32_t current_time = micros();
  
  // Main keyboard scan at 1kHz
  if (current_time - last_scan >= SCAN_INTERVAL_US) {
    last_scan = current_time;
    scanMatrix();
    scanPedals();
  }
  
  // Handle any serial commands from Pi (optional)
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "STATUS") {
      Serial.println("SCANNER_OK");
    } else if (command == "RESET") {
      InitialiseKeyStates();
      Serial.println("RESET_OK");
    }
  }
  
  // Small delay to prevent overwhelming the CPU
  delayMicroseconds(100);
}
