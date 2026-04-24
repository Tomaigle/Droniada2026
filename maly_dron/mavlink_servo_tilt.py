from pymavlink import mavutil
from gpiozero import Servo
import math
import time

# --- Configuration ---
# Serial Connection
CONNECTION_STRING = "/dev/serial0"
BAUD_RATE = 57600

# Servo Setup
SERVO_PIN = 4
# Using the wide-range tuned values we discussed earlier
servo = Servo(SERVO_PIN, min_pulse_width=1.0 / 1000, max_pulse_width=2.0 / 1000)

# Flight Logic
ROLL_THRESHOLD_DEG = 25.0  # Trigger if drone tilts more than 45 degrees left or right

# ---------------------

print("Initializing Servo to OPEN position...")
servo.min()  # Set servo to your "Open" or starting position
time.sleep(1)  # Give it a second to physically move

print(f"Connecting to ArduPilot on {CONNECTION_STRING}...")
master = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUD_RATE)

# 1. Wait for connection
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Target found! We have a connection.")

# 2. Wait for the drone to be ARMED
print("\n--- WAITING FOR ARM ---")
print("Waiting for you to arm the drone via Radio Transmitter...")

while True:
    # We listen to the 1Hz Heartbeat message, which contains the system status
    msg = master.recv_match(type="HEARTBEAT", blocking=True)

    # Check if the "SAFETY_ARMED" bit is flipped to 1 in the base_mode registry
    if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
        print("\n✅ DRONE IS ARMED! Starting telemetry monitor...")
        break

# 3. Drone is armed! Request fast Attitude Data
master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
    10,
    1,  # 10 Hz rate, Start sending
)

print(f"\n--- MONITORING ROLL (Threshold: {ROLL_THRESHOLD_DEG}°) ---")
servo_triggered = False

# 4. Telemetry Loop
try:
    while True:
        msg = master.recv_match(type="ATTITUDE", blocking=True)
        if msg:
            # MAVLink sends angles in radians. We convert to degrees to make it readable.
            roll_deg = math.degrees(msg.roll)

            # Print live data to the terminal (overwriting the same line)
            print(f"Live Roll: {roll_deg:>6.2f}°", end="\r")

            # Check if the absolute value (left or right tilt) exceeds the threshold
            if abs(roll_deg) > ROLL_THRESHOLD_DEG and not servo_triggered:
                print(f"\n\n🚨 THRESHOLD EXCEEDED: {roll_deg:.2f}°!")
                print("Activating Servo -> CLOSING...")

                # Fire the servo to its max position
                servo.max()
                servo_triggered = True

                # We don't break the loop here so we can still see telemetry,
                # but 'servo_triggered' prevents it from spamming the servo command.

except KeyboardInterrupt:
    print("\n\nUser terminated script. Releasing servo and closing connection...")
    servo.detach()  # Stop sending PWM to the servo so it doesn't jitter
    master.close()
