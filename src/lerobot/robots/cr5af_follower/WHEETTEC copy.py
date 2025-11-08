# motor_keyboard_control_kb.py
# Cross-platform keyboard-driven motor control over serial
# - 非阻塞键盘读取；在 Linux/macOS 无 stdin TTY 时回退到 /dev/tty
# - 支持一次性发送模式 (--once) 无需键盘
# - 协议：11B 帧 0x7B ... BCC 0x7D；BCC 为前9字节 XOR

import argparse
import os
import sys
import time
import serial

# ======== 按键读取（跨平台非阻塞，支持无 stdin TTY 的场景） ========
IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    import msvcrt

    def getch_nonblock(timeout=0.1):
        """
        Windows: 使用 msvcrt.kbhit/getch 非阻塞读取，timeout 为轮询时长。
        无按键返回 ""。
        """
        start = time.time()
        while (time.time() - start) < timeout:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    return ch.decode("utf-8", "ignore")
                except Exception:
                    return ""
            time.sleep(0.01)
        return ""

else:
    import termios, tty, select

    _TTY_FILE = None        # 持久化的 TTY 文件对象（stdin 或 /dev/tty）
    _WARNED_NO_TTY = False  # 只提示一次

    def _ensure_tty_file():
        """
        优先使用交互式 stdin；否则尝试打开 /dev/tty。
        成功返回带 fileno() 的文件对象；拿不到返回 None（禁用键盘，但不报错）。
        """
        global _TTY_FILE, _WARNED_NO_TTY
        if _TTY_FILE is not None:
            return _TTY_FILE

        # 1) stdin 可用则用之
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                _TTY_FILE = sys.stdin
                return _TTY_FILE
        except Exception:
            pass

        # 2) 回退到 /dev/tty
        try:
            _TTY_FILE = open("/dev/tty", "rb", buffering=0)
            return _TTY_FILE
        except Exception:
            if not _WARNED_NO_TTY:
                print("[WARN] No TTY available (stdin is not a tty and /dev/tty can't be opened). "
                      "Keyboard control is disabled in this session. "
                      "Run in a real terminal or use --once mode.",
                      flush=True)
                _WARNED_NO_TTY = True
            _TTY_FILE = None
            return None

    def getch_nonblock(timeout=0.1):
        """
        Unix: 非阻塞按键读取。
        - 若拿到 TTY，则在 raw 模式下等待 timeout 秒，读到1字节即返回字符；
        - 若没有 TTY，返回 ""，不抛异常（便于 --once 或后台模式）。
        """
        tty_file = _ensure_tty_file()
        if tty_file is None:
            time.sleep(timeout)
            return ""

        fd = tty_file.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            # 虽拿到 fd 但不可做 termios，视为无键盘
            time.sleep(timeout)
            return ""

        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([fd], [], [], timeout)
            if rlist:
                ch = os.read(fd, 1)
                try:
                    return ch.decode("utf-8", "ignore")
                except Exception:
                    return ""
            return ""
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ======== 协议与串口工具 ========
HEADER = 0x7B
TAIL   = 0x7D

def calculate_bcc(first_n_bytes: bytes) -> int:
    bcc = 0
    for b in first_n_bytes:
        bcc ^= b
    return bcc & 0xFF

def clamp_u16(val: int) -> int:
    return max(0, min(0xFFFF, int(val)))

def build_control_frame(dev_id: int, mode: int, direction: int,
                        microstep: int, angle_u16: int, speed_u16: int) -> bytearray:
    """
    11-byte frame:
    [0]=0x7B, [1]=id, [2]=mode, [3]=dir, [4]=microstep,
    [5]=angle_hi, [6]=angle_lo, [7]=speed_hi, [8]=speed_lo,
    [9]=bcc(first 9 bytes XOR), [10]=0x7D
    """
    angle_u16 = clamp_u16(angle_u16)
    speed_u16 = clamp_u16(speed_u16)
    frame = bytearray(11)
    frame[0]  = HEADER
    frame[1]  = dev_id & 0xFF
    frame[2]  = mode & 0xFF            # 0x02 位置控制；0x00 在本脚本用作“状态请求”
    frame[3]  = 1 if direction else 0  # 0/1
    frame[4]  = microstep & 0xFF
    frame[5]  = (angle_u16 >> 8) & 0xFF
    frame[6]  = angle_u16 & 0xFF
    frame[7]  = (speed_u16 >> 8) & 0xFF
    frame[8]  = speed_u16 & 0xFF
    frame[9]  = calculate_bcc(frame[:9])
    frame[10] = TAIL
    return frame

def build_status_request(dev_id: int) -> bytearray:
    """
    以同样的 11 字节帧格式构造“状态请求”：mode=0x00，其他字段清零。
    若设备协议不同，请按文档修改此函数。
    """
    return build_control_frame(dev_id=dev_id, mode=0x00,
                               direction=0, microstep=0x00,
                               angle_u16=0x0000, speed_u16=0x0000)

def frame_to_hex(frame: bytes) -> str:
    return " ".join(f"{b:02X}" for b in frame)

def open_serial(port: str, baud: int, timeout: float = 0.5) -> serial.Serial:
    return serial.Serial(port, baudrate=baud, timeout=timeout)

def send_and_maybe_read(ser: serial.Serial, frame: bytes, read_len: int = 0, wait: float = 0.05) -> bytes:
    """
    发送一帧；若 read_len>0 则按固定长度读取；否则读尽缓冲区。
    """
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    time.sleep(wait)
    if read_len > 0:
        return ser.read(read_len)
    n = ser.in_waiting
    return ser.read(n) if n > 0 else b""

# ======== 状态解析（9 字节） ========
def parse_status_9b(payload: bytes) -> dict:
    """
    [0]=addr,[1]=reached,[2]=speed_hi,[3]=speed_lo,
    [4]=angle_31_24,[5]=angle_23_16,[6]=angle_15_8,[7]=angle_7_0,[8]=RCC
    """
    if len(payload) != 9:
        raise ValueError(f"Expected 9 bytes, got {len(payload)}")
    addr = payload[0]
    reached = payload[1]
    speed = (payload[2] << 8) | payload[3]
    angle = (payload[4] << 24) | (payload[5] << 16) | (payload[6] << 8) | payload[7]
    rcc = payload[8]
    return {
        "address": addr,
        "position_reached": reached,
        "speed_u16": speed,
        "angle_u32": angle,
        "rcc": rcc,
    }

def print_status_dict(st: dict):
    print("—— 状态反馈 ——")
    print(f"设备地址    : {st['address']:d}")
    print(f"到位标志    : {st['position_reached']} (0=未到,1=到达)")
    print(f"当前速度(u16): {st['speed_u16']}")
    print(f"当前角度(u32): {st['angle_u32']}")
    print(f"RCC校验字节 : 0x{st['rcc']:02X}")

# ======== 主程序 ========
def main():
    parser = argparse.ArgumentParser(description="Keyboard-like motor control over serial (non-blocking).")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port, e.g., /dev/ttyACM0 or COM3")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--dev_id", type=int, default=0x01, help="Device ID byte")
    parser.add_argument("--microstep", type=lambda x: int(x, 0), default=0x20, help="Microstep byte (e.g., 0x20)")
    parser.add_argument("--init_angle", type=int, default=0, help="Initial angle (u16 units 0..65535)")
    parser.add_argument("--init_speed", type=int, default=100, help="Initial speed (u16 units 0..65535)")
    parser.add_argument("--read_status_len", type=int, default=9, help="Expected status length (bytes)")
    # 一次性发送模式（无键盘）
    parser.add_argument("--once", action="store_true",
                        help="One-shot: send control frame once and (optionally) read status, then exit")
    parser.add_argument("--angle", type=lambda x: int(x, 0), help="Angle u16 for --once (e.g., 0x1234 or 1000)")
    parser.add_argument("--speed", type=lambda x: int(x, 0), help="Speed u16 for --once")
    parser.add_argument("--dir",   type=int, choices=[0,1], help="Direction 0/1 for --once")
    parser.add_argument("--mode_once", type=lambda x: int(x,0), help="Mode byte for --once (default 0x02)")
    parser.add_argument("--status", action="store_true", help="With --once: request and print status after sending")
    args = parser.parse_args()

    angle = clamp_u16(args.init_angle)
    speed = clamp_u16(args.init_speed)
    direction = 1  # 1=顺时针, 0=逆时针
    mode = 0x02    # 0x02 位置控制；0x00 状态请求
    micro = args.microstep & 0xFF

    # 打开串口
    try:
        ser = open_serial(args.port, args.baud)
    except Exception as e:
        print(f"[ERR] 打开串口失败：{e}")
        sys.exit(1)

    # ---- 一次性发送模式：无需键盘/TTY ----
    if args.once:
        ang = clamp_u16(args.angle if args.angle is not None else angle)
        spd = clamp_u16(args.speed if args.speed is not None else speed)
        d   = args.dir if args.dir is not None else direction
        md  = args.mode_once if args.mode_once is not None else 0x02

        frame = build_control_frame(args.dev_id, md, d, micro, ang, spd)
        print(f"[ONCE] 发送控制帧: {frame_to_hex(frame)}")
        resp = send_and_maybe_read(ser, frame, read_len=0)
        if resp:
            print(f"[ONCE] 回读({len(resp)}B): {frame_to_hex(resp)}")

        if args.status:
            req = build_status_request(args.dev_id)
            print(f"[ONCE] 请求状态: {frame_to_hex(req)}")
            data = send_and_maybe_read(ser, req, read_len=args.read_status_len, wait=0.08)
            if not data:
                time.sleep(0.05)
                n = ser.in_waiting
                if n:
                    data = ser.read(n)
            if data:
                print(f"[ONCE] 状态原始({len(data)}B): {frame_to_hex(data)}")
                if len(data) == 9:
                    try:
                        st = parse_status_9b(data); print_status_dict(st)
                    except Exception as pe:
                        print(f"[WARN] 解析 9B 状态失败：{pe}")
            else:
                print("[ONCE] 未收到状态返回。")
        try:
            ser.close()
        except Exception:
            pass
        return

    # ---- 交互键控模式 ----
    print(f"已连接串口 {args.port} @ {args.baud}")
    if not IS_WINDOWS:
        # 触发一次 TTY 检查（若无 TTY 会提示 WARN，但不中断）
        try:
            _ = _ensure_tty_file()
        except NameError:
            pass

    def print_params():
        ctrl = build_control_frame(args.dev_id, mode, direction, micro, angle, speed)
        print("—— 当前参数 ——")
        print(f"mode={mode:#04x} (0=状态请求,2=位置控制), dir={direction}, micro=0x{micro:02X}")
        print(f"angle(u16)={angle}, speed(u16)={speed}")
        print(f"[控制帧] {frame_to_hex(ctrl)}")

    print("按键帮助: a/A d/D s/S w/W r m f t p，q=退出")
    print(" a/A: 角度-1 / -10     d/D: 角度+1 / +10")
    print(" s/S: 速度-1 / -10     w/W: 速度+1 / +10")
    print(" r  : 切换方向(0/1)    m  : 切换模式(0=状态请求,2=位置控制)")
    print(" f  : 发送当前控制帧   t  : 发送状态请求并解析打印")
    print(" p  : 打印当前参数与帧 q  : 退出")

    try:
        while True:
            try:
                c = getch_nonblock(timeout=0.2)
            except KeyboardInterrupt:
                c = "q"

            if not c:
                # 空闲时可做心跳、定时状态轮询等
                continue

            c = c[0]  # 仅取单个字符

            if c in ("q", "Q"):
                print("退出。")
                break

            elif c == "a":
                angle = clamp_u16(angle - 1);  print(f"angle={angle}")
            elif c == "A":
                angle = clamp_u16(angle - 10); print(f"angle={angle}")
            elif c == "d":
                angle = clamp_u16(angle + 1);  print(f"angle={angle}")
            elif c == "D":
                angle = clamp_u16(angle + 10); print(f"angle={angle}")
            elif c == "s":
                speed = clamp_u16(speed - 1);  print(f"speed={speed}")
            elif c == "S":
                speed = clamp_u16(speed - 10); print(f"speed={speed}")
            elif c == "w":
                speed = clamp_u16(speed + 1);  print(f"speed={speed}")
            elif c == "W":
                speed = clamp_u16(speed + 10); print(f"speed={speed}")
            elif c == "r":
                direction = 0 if direction == 1 else 1; print(f"dir={direction}")
            elif c == "m":
                mode = 0x00 if mode != 0x00 else 0x02; print(f"mode=0x{mode:02X}")
            elif c == "p":
                print_params()
            elif c == "f":
                frame = build_control_frame(args.dev_id, mode, direction, micro, angle, speed)
                print(f"发送控制帧: {frame_to_hex(frame)}")
                resp = send_and_maybe_read(ser, frame, read_len=0)
                if resp:
                    print(f"收到回读({len(resp)}): {frame_to_hex(resp)}")
                else:
                    print("无回读数据。")
            elif c == "t":
                req = build_status_request(args.dev_id)
                print(f"发送状态请求: {frame_to_hex(req)}")
                data = send_and_maybe_read(ser, req, read_len=args.read_status_len, wait=0.08)
                if not data:
                    # 也许设备返回不是固定长度，尝试读缓冲区
                    time.sleep(0.05)
                    n = ser.in_waiting
                    if n:
                        data = ser.read(n)
                if data:
                    print(f"原始返回({len(data)}B): {frame_to_hex(data)}")
                    if len(data) == 9:
                        try:
                            st = parse_status_9b(data); print_status_dict(st)
                        except Exception as pe:
                            print(f"[WARN] 解析 9B 状态失败：{pe}")
                    else:
                        print("[提示] 返回长度不是 9 字节，可能设备有不同协议或额外封装。")
                else:
                    print("未收到状态返回。")
            else:
                # 其它键忽略
                pass
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
