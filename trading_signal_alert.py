# -*- coding: utf-8 -*-
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from plyer import notification
import requests
from datetime import datetime
import threading
import queue
import pygame
import os
import logging
from dotenv import load_dotenv

class TradingSignalAlert:
    def __init__(self, config=None):
        """
        初始化交易信号提醒系统
        
        参数:
            config (dict): 配置字典，包含各种通知方式的参数
        """
        # 加载 .env 文件
        load_dotenv()
        # 默认配置
        self.config = {
            'sound': True,  # 启用声音提醒
            'desktop': True,  # 启用桌面通知
            'email': True,  # 启用邮件通知
            'sms': False,    # 启用短信通知
            'push': False,   # 启用推送通知
            
            # 声音配置
            'buy_sound': 'buy_sound.wav',  # 买入声音文件
            'sell_sound': 'sell_sound.wav', # 卖出声音文件
            
            # 邮件配置
            'email_sender': os.getenv('TRADING_ALERT_EMAIL_SENDER'),
            'email_password': os.getenv('TRADING_ALERT_EMAIL_PASSWORD'),
            'email_receiver': os.getenv('TRADING_ALERT_EMAIL_RECEIVER'),
            'smtp_server': os.getenv('TRADING_ALERT_SMTP_SERVER'),
            'smtp_port': int(os.getenv('TRADING_ALERT_SMTP_PORT')),
            
            # 短信配置 (需要短信服务提供商)
            'sms_api_key': os.getenv('TRADING_ALERT_SMS_API_KEY'),
            'sms_sender': 'TradingBot',
            'sms_receiver': os.getenv('TRADING_ALERT_SMS_RECEIVER'),
            
            # 推送配置 (如Pushover, Telegram等)
            'push_api_key': os.getenv('TRADING_ALERT_PUSHOVER_API_KEY'),
            'push_user_key': os.getenv('TRADING_ALERT_PUSHOVER_USER_KEY'),
            
            # 日志配置
            'log_file': 'trading_alerts.log'
        }
        
        # 更新自定义配置
        if config:
            self.config.update(config)
        
        # 初始化声音系统
        if self.config['sound']:
            try:
                pygame.mixer.init()
                # 检查声音文件是否存在，不存在则创建默认声音
                self._create_default_sounds()
            except Exception as e:
                logging.error(f"声音初始化失败: {e}")
                self.config['sound'] = False
        
        # 初始化日志系统
        logging.basicConfig(
            filename=self.config['log_file'],
            level=logging.INFO,
            encoding='utf-8',
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 消息队列
        self.alert_queue = queue.Queue()
        
        # 启动提醒处理线程
        self.alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.alert_thread.start()
    
    def _create_default_sounds(self):
        """创建默认声音文件（如果不存在）"""
        if not os.path.exists(self.config['buy_sound']):
            # 生成买入提示音
            self._generate_sound(800, 500, self.config['buy_sound'])
        
        if not os.path.exists(self.config['sell_sound']):
            # 生成卖出提示音
            self._generate_sound(500, 800, self.config['sell_sound'])
    
    def _generate_sound(self, freq1, freq2, filename):
        """生成简单的提示音（正弦波叠加）"""
        try:
            import wave
            import struct
            import math

            duration = 0.5  # 秒
            sample_rate = 44100
            samples = int(duration * sample_rate)

            # 生成两个正弦波并叠加
            wave_data = []
            for i in range(samples):
                t = i / sample_rate
                value = (
                    0.5 * math.sin(2 * math.pi * freq1 * t) +
                    0.5 * math.sin(2 * math.pi * freq2 * t)
                )
                # 限幅并转换为16位整数
                sample = int(max(-1.0, min(1.0, value)) * 32767)
                wave_data.append(sample)

            with wave.open(filename, 'w') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sample_rate)
                f.writeframes(b''.join(struct.pack('<h', sample) for sample in wave_data))

            logging.info(f"创建默认声音文件: {filename}")
        except Exception as e:
            logging.error(f"无法创建声音文件 {filename}: {e}")
    
    def send_alert(self, signal_type, message, chart = None):
        """
        发送交易信号提醒
        
        参数:
            signal_type (str): 'buy' 或 'sell'
            message (str): 提醒消息内容
        """
        # 将提醒加入队列
        self.alert_queue.put((signal_type, message, chart))
    
    def _process_alerts(self):
        """处理提醒队列的后台线程"""
        while True:
            try:
                signal_type, message, chart = self.alert_queue.get()
                
                # 记录日志
                logging.info(f"交易信号: {signal_type.upper()} - {message}")
                
                # 声音提醒
                if self.config['sound']:
                    self._play_sound(signal_type)
                
                # 桌面通知
                if self.config['desktop']:
                    self._show_desktop_notification(signal_type, message)
                
                # 邮件通知
                if self.config['email']:
                    threading.Thread(target=self._send_email, args=(signal_type, message, chart)).start()
                
                # 短信通知
                if self.config['sms']:
                    threading.Thread(target=self._send_sms, args=(signal_type, message)).start()
                
                # 推送通知
                if self.config['push']:
                    threading.Thread(target=self._send_push, args=(signal_type, message)).start()
                
                # 标记任务完成
                self.alert_queue.task_done()
                
                # 短暂延迟以避免通知风暴
                time.sleep(1)
            
            except Exception as e:
                logging.error(f"处理提醒时出错: {e}")
    
    def _play_sound(self, signal_type):
        """播放声音提醒"""
        try:
            sound_file = self.config['buy_sound'] if signal_type == 'buy' else self.config['sell_sound']
            
            # 使用pygame播放声音
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            sound = pygame.mixer.Sound(sound_file)
            channel = sound.play()
            # 等待声音播放完成
            while channel.get_busy():
                time.sleep(0.1)
        except Exception as e:
            logging.error(f"播放声音失败: {e}")
    
    def _show_desktop_notification(self, signal_type, message):
        """显示桌面通知"""
        try:
            title = "买入信号" if signal_type == 'buy' else "卖出信号"
            notification.notify(
                title=f"交易提醒: {title}",
                message=message,
                app_name="ETF交易策略",
                timeout=10  # 显示10秒
            )
        except Exception as e:
            logging.error(f"桌面通知失败: {e}")
    
    def _send_email(self, signal_type, message, chart = None):
        """发送邮件通知"""
        try:
            zscore_cid = 'zscore_chart@trading'
            subject = f"交易信号: {'买入' if signal_type == 'buy' else '卖出'}"
            body = f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .chart-container {{
                        margin-bottom: 30px;
                        border: 1px solid #e0e0e0;
                        border-radius: 5px;
                        padding: 10px;
                        background-color: white;
                    }}
                    .chart-title {{
                        font-size: 16px;
                        font-weight: bold;
                        margin-bottom: 10px;
                        color: #2c3e50;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        display: block;
                        margin: 0 auto;
                    }}
                </style>
            </head>
            <body>
                <h2>{subject}</h2>
                <p>{message}</p>
                <p>时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <div class="chart-container">
                    <div class="chart-title">Z-Score指标与动态阈值</div>
                    <img src="cid:{zscore_cid}" alt="Z-Score走势">
                </div>
            </body>
            </html>
            """
            email_receivers = self.config['email_receiver'].split(',')
            msg = MIMEMultipart('related')
            msg['Subject'] = subject
            msg['From'] = self.config['email_sender']
            msg['To'] = ", ".join(email_receivers)
            
            # 文字内容
            msg_body = MIMEText(body, 'html')
            msg.attach(msg_body)
            # 添加Z-Score图表
            if chart is not None:
                msg_image = MIMEImage(chart)
                msg_image.add_header('Content-ID', f'<{zscore_cid}>')
                msg.attach(msg_image)
            
            # 检查SMTP服务器地址是否正确
            smtp_server = self.config.get('smtp_server', '').strip()
            smtp_port = self.config.get('smtp_port', 587)
            if not smtp_server:
                logging.error("SMTP服务器地址未配置")
                return

            try:
                with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(self.config['email_sender'], self.config['email_password'])
                    server.send_message(msg, msg['From'], email_receivers)
                logging.info(f"邮件通知已发送")
            except smtplib.SMTPResponseException as e:
                # 邮件已发送但返回出现异常
                logging.info(f"邮件通知已发送，但返回出现异常: {e}")
            except Exception as e:
                logging.error(f"连接SMTP服务器失败: {e}")
        except Exception as e:
            logging.error(f"发送邮件失败: {e}")
    
    def _send_sms(self, signal_type, message):
        """发送短信通知（需要短信服务提供商）"""
        try:
            # 这里使用Twilio作为示例，您需要替换为实际使用的短信服务API
            signal_text = "买入信号" if signal_type == 'buy' else "卖出信号"
            full_message = f"{signal_text}: {message}"
            
            # 示例：使用Twilio发送短信
            # from twilio.rest import Client
            # client = Client(self.config['sms_api_key'], 'your_auth_token')
            # client.messages.create(
            #     body=full_message,
            #     from_=self.config['sms_sender'],
            #     to=self.config['sms_receiver']
            # )
            
            # 实际实现需要根据您的短信服务提供商API进行修改
            logging.warning("短信功能未实现，请配置短信服务提供商API")
        except Exception as e:
            logging.error(f"发送短信失败: {e}")
    
    def _send_push(self, signal_type, message):
        """发送推送通知（使用Pushover作为示例）"""
        try:
            signal_text = "买入信号" if signal_type == 'buy' else "卖出信号"
            title = f"交易提醒: {signal_text}"
            
            # Pushover API
            url = "https://api.pushover.net/1/messages.json"
            data = {
                "token": self.config['push_api_key'],
                "user": self.config['push_user_key'],
                "message": message,
                "title": title,
                "sound": "cashregister" if signal_type == 'buy' else "siren"
            }
            
            response = requests.post(url, data=data)
            if response.status_code != 200:
                logging.error(f"推送通知失败: {response.text}")
            else:
                logging.info("推送通知已发送")
        except Exception as e:
            logging.error(f"发送推送失败: {e}")