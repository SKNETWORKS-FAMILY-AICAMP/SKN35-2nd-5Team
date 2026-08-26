from pathlib import Path

import streamlit.components.v1 as components

wheel_picker_component = components.declare_component(
    "wheel_picker",
    path=str(Path(__file__).parent / "streamlit_components" / "wheel_picker"),
)
