from src.app.main import app; [print(f"{route.path} {route.methods}") for route in app.routes]
