from ui.animation import AnimatedValue
import time

value = AnimatedValue(
    initial=0,
    speed=0.1,
)

value.set_target(1)

for _ in range(50):
    print(round(value.update(), 3))
    time.sleep(0.03)