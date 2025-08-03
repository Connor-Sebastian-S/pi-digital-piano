# Power Service  
sudo chmod +x /usr/local/bin/safe_shutdown.py  
sudo wget -O /etc/systemd/system/safe-shutdown.service https://raw.githubusercontent.com/Connor-Sebastian-S/pi-digital-piano/refs/heads/main/safe-shutdown.service 
sudo systemctl enable safe-shutdown.service  

# Monitor Service  
chmod +x /home/admin/piano/start_monitoring.sh  
sudo wget -O /etc/systemd/system/piano-monitor.service https://raw.githubusercontent.com/Connor-Sebastian-S/pi-digital-piano/refs/heads/main/piano-monitor.service 
sudo systemctl enable piano-monitor.service  

# Final Activation  
sudo systemctl daemon-reload  
sudo systemctl start safe-shutdown.service  
sudo systemctl start piano-monitor.service  
