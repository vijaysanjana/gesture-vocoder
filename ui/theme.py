from dataclasses import dataclass


Color = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    background_tint: Color = (24, 10, 38)
    skeleton: Color = (255, 220, 120)
    fingertip: Color = (255, 90, 220)
    pinch: Color = (255, 80, 170)
    text: Color = (245, 245, 245)
    muted_text: Color = (180, 170, 190)
    panel: Color = (35, 20, 50)
    panel_border: Color = (120, 90, 150)
    glow: Color = (255, 100, 220)