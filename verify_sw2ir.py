
import sys
# Set up paths if needed
try:
    from Sw2IR import Sw2IR
except:
    sys.path.append("/Users/luna/Documents/Research/Chauvet/Sw2IR")
    from Sw2IR import Sw2IR
    
from PySide6.QtWidgets import QApplication

print("Import successful")
try:
    # Enable High DPI
    if hasattr(QApplication, 'setAttribute'):
        from PySide6.QtCore import Qt
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    app = QApplication(sys.argv)
    window = Sw2IR()
    print("Sw2IR instantiated successfully")
    
    if window.process_btn and window.norm_check and window.magic_toggle:
        print("Widgets created successfully")
        
    print("Verification passed")
    sys.exit(0)
    
except Exception as e:
    print(f"App failed: {e}")
    sys.exit(1)
