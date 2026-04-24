from pymavlink import mavutil
import time

CONNECTION_STRING = "/dev/serial0"
BAUD_RATE = 57600

print(f"Connecting to ArduPilot on {CONNECTION_STRING}...")
master = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUD_RATE)

# 1. Wait for connection
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Target found! We have a connection.")

# 2. Request Telemetry Data
master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
    10,
    1,
)

# 3. Send the ARM Command
print("\n--- ATTEMPTING TO ARM ---")
# command_long_send parameters: target_system, target_component, command, confirmation, param1, param2-7
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,  # Confirmation
    1,  # Param 1: 1 means ARM, (0 would mean DISARM)
    0,
    0,
    0,
    0,
    0,
    0,  # Params 2-7 are not used for this command
)

# 4. Wait for ArduPilot to reply (Acknowledge)
print("Waiting for flight controller response...")
# We wait up to 3 seconds for a COMMAND_ACK message
ack_msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=3.0)

if ack_msg:
    # result == 0 means MAV_RESULT_ACCEPTED
    if ack_msg.result == 0:
        print("✅ SUCCESS: Drone is ARMED! Motors should be spinning.")
    else:
        print(f"❌ REJECTED: ArduPilot refused to arm. Result code: {ack_msg.result}")
        print("Reason: You likely failed a Pre-Arm Check (e.g., no GPS, not level).")
else:
    print("⚠️ No acknowledgment received. Command may have been lost.")

print("\n--- Reading Telemetry (Press Ctrl+C to stop) ---")

# 5. Continue reading telemetry loop
try:
    while True:
        msg = master.recv_match(type="ATTITUDE", blocking=True)
        if msg:
            print(
                f"Roll: {msg.roll:.3f} | Pitch: {msg.pitch:.3f} | Yaw: {msg.yaw:.3f}",
                end="\r",
            )

except KeyboardInterrupt:
    print("\n\nDisarming and closing connection...")
    # Send Disarm Command before quitting
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # Param 1 is 0 to DISARM
    )
    master.close()
