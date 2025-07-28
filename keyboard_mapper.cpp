#include <Arduino.h>

const int ROW_PINS[] = {15, 16, 17, 18, 8, 3, 46, 9};  // T0–T7
const int NUM_ROWS = 8;

// Combined MK and BR pins for 22 columns
const int MK_PINS[] = {
  10, 11, 12, 13, 14,   // MK0–MK4
  1, 2, 42, 41, 39,     // MK5–MK9
  40,                  // MK10
};

const int BR_PINS[] = {
  38, 37, 36, 35, 45,   // BR0–BR4
  48, 47, 4, 5, 6,      // BR5–BR9
  7,                   // BR10
};

const int NUM_MK = sizeof(MK_PINS) / sizeof(int);
const int NUM_BR = sizeof(BR_PINS) / sizeof(int);

// Track press detection
bool last_state[NUM_MK][NUM_ROWS] = {false};

void setup() {
  Serial.begin(115200);

  // Setup row outputs
  for (int i = 0; i < NUM_ROWS; i++) {
    pinMode(ROW_PINS[i], OUTPUT);
    digitalWrite(ROW_PINS[i], HIGH);
  }

  // Setup MK and BR inputs
  for (int i = 0; i < NUM_MK; i++) {
    pinMode(MK_PINS[i], INPUT_PULLUP);
  }
  for (int i = 0; i < NUM_BR; i++) {
    pinMode(BR_PINS[i], INPUT_PULLUP);
  }

  Serial.println("🎹 Key Matrix Logger Ready");
  Serial.println("Press keys in order (A0 to C8). Each press will log MK+BR+row.");
}

void loop() {
  for (int row = 0; row < NUM_ROWS; row++) {
    // Activate current row
    for (int i = 0; i < NUM_ROWS; i++) {
      digitalWrite(ROW_PINS[i], (i == row) ? LOW : HIGH);
    }

    delayMicroseconds(10);  // Allow line to settle

    for (int mk = 0; mk < NUM_MK; mk++) {
      bool mk_pressed = (digitalRead(MK_PINS[mk]) == LOW);

      if (mk_pressed && !last_state[mk][row]) {
        // Try to find matching BR for the same row
        int br_match = -1;
        for (int br = 0; br < NUM_BR; br++) {
          if (digitalRead(BR_PINS[br]) == LOW) {
            br_match = br;
            break;
          }
        }

        Serial.print("KEY DETECTED: ");
        Serial.print("Row="); Serial.print(row);
        Serial.print(", MK="); Serial.print(mk);
        if (br_match >= 0) {
          Serial.print(", BR="); Serial.print(br_match);
        } else {
          Serial.print(", BR=none");
        }
        Serial.println();

        // Wait for key to release
        while (digitalRead(MK_PINS[mk]) == LOW) {
          delay(10);
        }
      }

      last_state[mk][row] = mk_pressed;
    }
  }

  delay(1);
}
