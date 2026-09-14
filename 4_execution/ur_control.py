import socket
import time

ROBOT_IP = "192.168.1.100"
DASHBOARD_PORT = 29999

def send_dashboard_command(ip, port, command):
    # Create a standard TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        s.connect((ip, port))
        
        # The dashboard server sends a welcome message upon connection; receive and discard it
        s.recv(1024)
        
        # Send the command (must be encoded to bytes and end with a newline)
        s.sendall(f"{command}\n".encode('utf-8'))
        
        # Receive the response
        response = s.recv(1024).decode('utf-8')
        return response.strip()

# Pause the robot
print("Sending pause command...")
response = send_dashboard_command(ROBOT_IP, DASHBOARD_PORT, "pause")
print(f"Robot replied: {response}")

time.sleep(3) # Wait for 3 seconds

# Play/resume the robot
print("Sending play command...")
response = send_dashboard_command(ROBOT_IP, DASHBOARD_PORT, "play")
print(f"Robot replied: {response}")




import dashboard_client # Included when you pip install ur_rtde
import time

ROBOT_IP = "192.168.1.100"

# Initialize and connect to the dashboard server
db_client = dashboard_client.DashboardClient(ROBOT_IP)
db_client.connect()

# Pause the running program (stops physical motion)
print("Pausing robot...")
db_client.pause()

time.sleep(3) # Wait 3 seconds

# Resume the running program (resumes physical motion)
print("Playing robot...")
db_client.play()

# Clean up connection
db_client.disconnect()