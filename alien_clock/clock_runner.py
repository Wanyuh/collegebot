import sys
import threading
from datetime import datetime
from playsound import playsound
from PyQt5.QtWidgets import (QApplication, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout)
from PyQt5.QtCore import QTimer
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class AlienClock(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        # Initialize Earth and alien time bases
        self.earth_base = datetime(1970, 1, 1, 0, 0, 0)  # Earth time
        self.alien_base = {
            'year': 2804,
            'month': 18,
            'day': 31,
            'hour': 2,
            'minute': 2,
            'second': 88
        }
        self.alien_month_days = [44, 42, 48, 40, 48, 44, 40, 44, 42,
                                 40, 40, 42, 44, 48, 42, 40, 44, 38]

        # Define alien time constants
        self.alien_seconds_per_minute = 90
        self.alien_minutes_per_hour = 90
        self.alien_hours_per_day = 36
        self.alien_months_per_year = 18

        # One alien second is 0.5 Earth seconds
        self.alien_to_earth_ratio = 0.5

        # Alarm variable
        self.alarm_time = None

        # Start updating the clocks
        self.update_clocks()

    def initUI(self):
        self.setWindowTitle("Alien and Earth Clock")

        # Create layout
        layout = QVBoxLayout()
        clock_layout = QHBoxLayout()
        earth_layout = QVBoxLayout()
        alien_layout = QVBoxLayout()

        # Earth clock label
        self.earth_clock_label = QLabel("Earth Time", self)
        self.earth_time_label = QLabel("", self)
        self.earth_time_label.setStyleSheet("font-size: 48px; color: blue;")

        # Alien clock label
        self.alien_clock_label = QLabel("Alien Time", self)
        self.alien_time_label = QLabel("", self)
        self.alien_time_label.setStyleSheet("font-size: 48px; color: green;")

        # Add Earth clock components to the Earth layout
        earth_layout.addWidget(self.earth_clock_label)
        earth_layout.addWidget(self.earth_time_label)

        # Add Alien clock components to the Alien layout
        alien_layout.addWidget(self.alien_clock_label)
        alien_layout.addWidget(self.alien_time_label)

        # Add the clock layouts to the main clock layout
        clock_layout.addLayout(earth_layout)
        clock_layout.addLayout(alien_layout)

        # Create a vertical layout for the alarm section
        self.alarm_label = QLabel("Set Alarm (hh:mm:ss)", self)
        self.alarm_entry = QLineEdit(self)
        self.alarm_entry.setText("25:72:84")  # Example alien time
        self.alarm_button = QPushButton("Set Alarm", self)
        self.alarm_button.clicked.connect(self.set_alarm)

        # Add alarm components to the layout
        layout.addLayout(clock_layout)  # Add the clock layout first
        layout.addWidget(self.alarm_label)  # Add the alarm label
        layout.addWidget(self.alarm_entry)  # Add the alarm entry
        layout.addWidget(self.alarm_button)  # Add the alarm button

        self.setLayout(layout)

    def calculate_alien_time(self, earth_time):
        # Calculate alien time from Earth time
        earth_elapsed_seconds = (earth_time - self.earth_base).total_seconds()
        alien_elapsed_seconds = earth_elapsed_seconds / self.alien_to_earth_ratio

        # Convert alien seconds into alien time (year, month, day, hour, minute, second)
        alien_year = self.alien_base['year']
        alien_month = self.alien_base['month']
        alien_day = self.alien_base['day']
        alien_hour = self.alien_base['hour']
        alien_minute = self.alien_base['minute']
        alien_second = self.alien_base['second']

        alien_second += alien_elapsed_seconds
        alien_minute += alien_second // self.alien_seconds_per_minute
        alien_second %= self.alien_seconds_per_minute
        alien_hour += alien_minute // self.alien_minutes_per_hour
        alien_minute %= self.alien_minutes_per_hour
        alien_day += alien_hour // self.alien_hours_per_day
        alien_hour %= self.alien_hours_per_day

        while alien_day > self.alien_month_days[alien_month - 1]:
            alien_day -= self.alien_month_days[alien_month - 1]
            alien_month += 1
            if alien_month > self.alien_months_per_year:
                alien_month = 1
                alien_year += 1

        return alien_year, alien_month, alien_day, int(alien_hour), int(alien_minute), int(alien_second)

    def update_clocks(self):
        # Update both Earth and alien clocks
        current_earth_time = datetime.now()
        alien_time = self.calculate_alien_time(current_earth_time)

        # Update Earth time label
        self.earth_time_label.setText(current_earth_time.strftime('%H:%M:%S'))

        # Update Alien time label
        alien_time_str = f"{alien_time[3]:02}:{alien_time[4]:02}:{int(alien_time[5]):02}"
        self.alien_time_label.setText(alien_time_str)

        # Check if alarm should sound
        if self.alarm_time and alien_time_str == self.alarm_time:
            self.sound_alarm()

        # Schedule to update again after 500 ms
        QTimer.singleShot(500, self.update_clocks)

    def set_alarm(self):
        # Set the alarm based on user input
        self.alarm_time = self.alarm_entry.text()
        logger.info(f"Alarm set for {self.alarm_time} in alien time.")

    def sound_alarm(self):
        logger.info("Alarm ringing!")
        sound_path = os.path.join(os.path.dirname(__file__), "alarm_sound.wav")
        threading.Thread(target=playsound, args=(sound_path,)).start()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    clock_app = AlienClock()
    clock_app.resize(400, 200)
    clock_app.show()
    sys.exit(app.exec_())
