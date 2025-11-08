import serial
import time

def to_hex_str(data: bytes) -> str:
    """安全地把任意 bytes 打印成 '01 02 0A ...' 的十六进制字符串。"""
    return " ".join(f"{b:02X}" for b in data)

def rcc_xor8(data: bytes) -> int:
    """前 8 字节异或求 RCC（状态帧长度为9时）。"""
    b = 0
    for x in data[:8]:
        b ^= x
    return b & 0xFF

def parse_status_9b(data: bytes) -> dict:
    """
    解析 9 字节状态帧：
    [0]=addr, [1]=reached, [2]=speed_hi, [3]=speed_lo,
    [4]=angle31..24, [5]=angle23..16, [6]=angle15..8, [7]=angle7..0, [8]=RCC
    返回字典，并包含 RCC 校验是否通过。
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes or bytearray")
    if len(data) != 9:
        raise ValueError(f"expected 9 bytes, got {len(data)}")

    addr    = data[0]
    reached = data[1]
    speed   = (data[2] << 8) | data[3]
    angle   = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
    rcc     = data[8]
    rcc_calc = rcc_xor8(data)

    return {
        "raw_hex": to_hex_str(data),
        "address": addr,
        "position_reached": reached,        # 0/1（若不是0/1，说明协议或读帧边界可能不一致）
        "speed_u16": speed,
        "angle_u32": angle,
        "rcc": rcc,
        "rcc_calc": rcc_calc,
        "rcc_ok": (rcc == rcc_calc),
    }

def print_status_9b(data: bytes) -> None:
    """友好打印解析结果。"""
    try:
        st = parse_status_9b(data)
    except Exception as e:
        print(f"[PARSE ERROR] {e}")
        return

    print(f"RAW (hex): {st['raw_hex']}")
    print(f"Address   : {st['address']}")
    print(f"Reached   : {st['position_reached']} (0=not reached, 1=reached)")
    print(f"Speed(u16): {st['speed_u16']}")
    print(f"Angle(u32): {st['angle_u32']}")
    print(f"RCC       : 0x{st['rcc']:02X} (calc=0x{st['rcc_calc']:02X})  OK={st['rcc_ok']}")

grip_angle = 0 # 1800  # 设定夹爪目标角度，单位为度
motor_command = bytearray(11)
# 填充每个字节的值
motor_command[0] = 0x7B      # 帧头 0x7B
motor_command[1] = 0x01      # 控制 ID 0x01
motor_command[2] = 0x02      # 控制模式 0x02 (位置控制模式)
motor_command[3] = 0x00      # 转向 1 (顺时针转动)
motor_command[4] = 0x20      # 步进电机细分值 0x20 (32 细分)
# 协议要求：角度放大10倍后拆分高低8位
# 例如 1872° -> 18720 -> 0x4920，高8位0x49，低8位0x20
grip_angle_int = int(grip_angle * 10) & 0xFFFF
motor_command[5] = (grip_angle_int >> 8) & 0xFF   # 角度高8位
motor_command[6] = grip_angle_int & 0xFF          # 角度低8位
# motor_command[7] = 0x00      # 转速数据的高 8 位
# motor_command[8] = 0x04      # 转速数据的低 8 位 10rad/s
motor_command[7] = 0x01      # 转速数据的高 8 位
motor_command[8] = 0x2c      # 转速数据的低 8 位 30rad/s
motor_command[9] = 0x00      # BCC 校验位，待计算
motor_command[10] = 0x7D     # 帧尾 0x7D
# 计算 BCC 校验位（前面 9 个字节的异或和）
def calculate_bcc(data):
    bcc = 0
    for byte in data:
        bcc ^= byte  # 对每个字节进行异或运算
    return bcc
motor_command[9] = calculate_bcc(motor_command[:9])

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.05)  # 20hz
# ser.write(bytearray([0x7b, 0x01, 0x02, 0x00, 0x20, 0x46, 0x50, 0x01, 0x2c, 0x63, 0x7d]))  # 张开 30rad/s 1800（原始指令，已注释）
ser.write(bytearray([0x7b, 0x01, 0x02, 0x01, 0x20, 0x46, 0x50, 0x01, 0x2c, 0x62, 0x7d]))  # 收缩 30rad/s 1800
# ser.write(motor_command)                                                                  # 发送计算好的指令
# ser.write(bytearray([0x7b, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7a, 0x7d]))  # 请求状态
# recv_data = ser.read(9)  # 读取数据
# print("Received data:", recv_data) 
# print_status_9b(recv_data)  # 解析并打印状态
ser.close()

# ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.05) # 20hz
# start_time = time.time()
# recv_data = ser.read(9)  # 读取数据
# print("Time elapsed:", time.time() - start_time)
# print("Received data:", recv_data)
# print_status_9b(recv_data)  # 解析并打印状态
# while True:
#     recv_data = ser.read(9)  # 读取数据
#     print("Received data:", recv_data)
#     print_status_9b(recv_data)  # 解析并打印状态
    # if ser.in_waiting > 0:  # 检查是否有数据可读
    #     recv_data = ser.read(9)  # 读取数据
    #     print("Received data:", recv_data)
    #     print_status_9b(recv_data)  # 解析并打印状态
    # else:
    #     continue
    #     break
# ser.close()



# ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)

# try:
#     while True:
#         # 先看缓冲里有没有至少 9 字节
#         if ser.in_waiting >= 9:
#             recv_data = ser.read(9)
#             if len(recv_data) == 9:
#                 print("Received:", recv_data.hex(" "))
#                 print_status_9b(recv_data)
#             else:
#                 # 没够 9 字节就丢到下次（也可以把残余拼回 buffer）
#                 pass
#         else:
#             # 没数据就小睡一会，别空转（Debug 能读到就是因为有“延时”）
#             time.sleep(0.02)
# except KeyboardInterrupt:
#     pass
# finally:
#     ser.close()

# ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.2)
# time.sleep(0.1)  # 等待数据返回
# while True:
#     recv_data = ser.read(9)   # 尝试一次读取 9B
#     if recv_data:
#         print("Received:", recv_data)
#         print_status_9b(recv_data)
#     else:
#         # 没有读到，继续下一次循环
#         continue


# import serial
# import time

# HEADER = 0x7B
# TAIL   = 0x7D

# def bcc_xor(first9):
#     b = 0
#     for x in first9:
#         b ^= x
#     return b & 0xFF

# def build_move(dir_=1, micro=0x20, angle=0x4650, speed=0x012C, dev_id=0x01, mode=0x02):
#     frame = bytearray(11)
#     frame[0]  = HEADER
#     frame[1]  = dev_id
#     frame[2]  = mode          # 0x02=位置控制
#     frame[3]  = 1 if dir_ else 0
#     frame[4]  = micro & 0xFF
#     frame[5]  = (angle >> 8) & 0xFF
#     frame[6]  = angle & 0xFF
#     frame[7]  = (speed >> 8) & 0xFF
#     frame[8]  = speed & 0xFF
#     frame[9]  = bcc_xor(frame[:9])
#     frame[10] = TAIL
#     return frame

# def build_status_req(dev_id=0x01):
#     first9 = [0x7B, dev_id, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]  # 你给的格式
#     frame = bytearray(first9 + [bcc_xor(first9), 0x7D])
#     return frame

# def read_exact(ser, n, timeout_s=0.05):
#     """在给定小超时内尽量读满 n 字节；读不到就返回当前拿到的"""
#     end = time.time() + timeout_s
#     buf = bytearray()
#     while len(buf) < n and time.time() < end:
#         chunk = ser.read(n - len(buf))
#         if chunk:
#             buf.extend(chunk)
#         else:
#             # 无数据，稍等一下
#             time.sleep(0.001)
#     return bytes(buf)

# def parse_status_9b(data9):
#     # [0]=addr, [1]=reached, [2]=speed_hi, [3]=speed_lo,
#     # [4]=angle31..24, [5]=angle23..16, [6]=angle15..8, [7]=angle7..0, [8]=RCC
#     if len(data9) != 9:
#         return None
#     addr = data9[0]
#     reached = data9[1]
#     speed = (data9[2] << 8) | data9[3]
#     angle = (data9[4] << 24) | (data9[5] << 16) | (data9[6] << 8) | data9[7]
#     rcc = data9[8]
#     return dict(addr=addr, reached=reached, speed=speed, angle=angle, rcc=rcc)

# def hexs(b):
#     return " ".join(f"{x:02X}" for x in b)

# def main():
#     # 串口建议用较短超时，便于高频轮询
#     ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.02)  # <= 20ms 小超时
#     try:
#         # 1) 发运动指令
#         move = build_move(dir_=1, micro=0x20, angle=0x4650, speed=0x012C, dev_id=0x01, mode=0x02)
#         ser.reset_input_buffer()  # 清掉旧垃圾数据，避免把上次残留当作这次的状态
#         ser.write(move)
#         ser.flush()
#         print("SEND MOVE:", hexs(move))

#         # 2) 周期性发“状态请求”，边发边读
#         poll_interval = 0.05  # 50ms
#         t0 = time.time()
#         timeout_total = 5.0   # 最多等 5s（按需调整）
#         last_print = 0.0

#         status_req = build_status_req(dev_id=0x01)

#         while True:
#             # 2.1 发送状态请求
#             ser.write(status_req)
#             ser.flush()

#             # 2.2 读取最多 9 字节（有些控制器一个请求只回一帧）
#             data = read_exact(ser, 9, timeout_s=ser.timeout or 0.05)

#             if len(data) == 9:
#                 st = parse_status_9b(data)
#                 if st:
#                     # 节流打印：每 100ms 打一行
#                     if time.time() - last_print > 0.10:
#                         print(f"STATUS raw={hexs(data)} | addr={st['addr']} reached={st['reached']} "
#                               f"speed={st['speed']} angle={st['angle']} rcc=0x{st['rcc']:02X}")
#                         last_print = time.time()
#                     # 到位就退出
#                     if st["reached"] == 1:
#                         print("Reached target. Done.")
#                         break
#             else:
#                 # 没回满 9 字节，可能设备还没处理完，稍等再发下一次请求
#                 pass

#             if time.time() - t0 > timeout_total:
#                 print("Timeout waiting for reach flag; stop polling.")
#                 break

#             time.sleep(poll_interval)

#     finally:
#         ser.close()

# if __name__ == "__main__":
#     main()
