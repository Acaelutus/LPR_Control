import serial
import time

# Параметры для подключения к последовательному порту
SERIAL_PORT = 'COM6'  # Замените на нужный порт, где подключен ваш контроллер
BAUD_RATE = 115200  # Должен совпадать с настройками контроллера
RELAY_COMMAND = "OPEN_RELAY"  # Команда для активации реле

def send_command_to_controller(command):
    try:
        # Открываем соединение с последовательным портом
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            # Отправляем команду на контроллер
            ser.write((command + '\n').encode())  # Добавляем символ новой строки для корректной обработки
            print(f"Команда {command} отправлена.")
            time.sleep(1)  # Ждем 1 секунду
            
            # Читаем ответ от контроллера
            while ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(f"Ответ от контроллера: {response}")
                
    except serial.SerialTimeoutException as e:
        print(f"Ошибка тайм-аута при отправке команды: {e}")
    except serial.SerialException as e:
        print(f"Ошибка последовательного порта: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    # Отправляем команду для включения реле
    send_command_to_controller(RELAY_COMMAND)
