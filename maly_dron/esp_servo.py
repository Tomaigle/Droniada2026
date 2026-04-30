import serial
import time

SERIAL_PORT = "/dev/ttyS0"
BAUD_RATE = 9600
STOP_VALUE = 90


def send_command(ser, command):
    ser.write((command + "\n").encode())
    print(f"Sent: {command}")
    timeout = time.time() + 2
    while time.time() < timeout:
        if ser.in_waiting:
            response = ser.readline().decode().strip()
            print(f"ESP32: {response}")
            return response
    print("Warning: No response")
    return None


def set_speed(ser, value):
    """
    value: 0-180
      90  = stop
      91-180 = forward (faster toward 180)
      89-0   = reverse (faster toward 0)
    """
    value = max(0, min(180, int(value)))
    return send_command(ser, f"SPEED:{value}")


def stop(ser):
    return send_command(ser, "STOP")


def forward(ser, speed_percent):
    """speed_percent: 0-100"""
    value = STOP_VALUE + int(speed_percent / 100 * 90)
    return set_speed(ser, value)


def reverse(ser, speed_percent):
    """speed_percent: 0-100"""
    value = STOP_VALUE - int(speed_percent / 100 * 90)
    return set_speed(ser, value)


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("Continuous Servo Controller")
        print("Commands:")
        print("  f <0-100>  → forward at % speed  (e.g. 'f 50')")
        print("  r <0-100>  → reverse at % speed  (e.g. 'r 75')")
        print("  s <0-180>  → raw servo value")
        print("  stop       → halt servo")
        print("  q          → quit\n")

        while True:
            user_input = input("> ").strip().lower()

            if user_input == "q":
                stop(ser)
                break
            elif user_input == "stop":
                stop(ser)
            elif user_input.startswith("f "):
                pct = float(user_input[2:])
                forward(ser, pct)
            elif user_input.startswith("r "):
                pct = float(user_input[2:])
                reverse(ser, pct)
            elif user_input.startswith("s "):
                val = int(user_input[2:])
                set_speed(ser, val)
            else:
                print("Unknown command. Use f/r/stop/s/q")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
