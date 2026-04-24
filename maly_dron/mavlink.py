from pymavlink import mavutil
import time

# --- Configuration ---
# /dev/serial0 is the primary hardware UART on a Raspberry Pi
CONNECTION_STRING = "/dev/serial0"
BAUD_RATE = 57600

print(f"Connecting to ArduPilot on {CONNECTION_STRING} at {BAUD_RATE} baud...")

# Start the connection
master = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUD_RATE)

# 1. Wait for a heartbeat
# This is crucial. It tells us the physical connection is working and ArduPilot is talking.
print("Waiting for heartbeat from Matek F405...")
master.wait_heartbeat()
print(
    f"Target found! System ID: {master.target_system}, Component ID: {master.target_component}"
)

# 2. Request continuous data stream
# By default, ArduPilot might not send the specific data we want fast enough.
# We request the 'EXTRA1' stream (which includes ATTITUDE) at 10Hz.
master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
    10,  # Request 10 Hz rate
    1,  # Start sending
)

print("Reading Live Attitude Data (Press Ctrl+C to stop)...")
print("-" * 50)

# 3. Main Loop: Read and print incoming MAVLink messages
try:
    while True:
        # We ask pymavlink to pull only the 'ATTITUDE' message from the stream
        msg = master.recv_match(type="ATTITUDE", blocking=True)

        if msg:
            # The data comes in radians, so it's a bit hard to read.
            # We can print the raw data, or you can import 'math' to convert to degrees.
            print(f"Roll: {msg.roll:.3f} | Pitch: {msg.pitch:.3f} | Yaw: {msg.yaw:.3f}")

except KeyboardInterrupt:
    print("\nClosing connection...")
    master.close()
