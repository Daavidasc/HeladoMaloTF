import cv2
import numpy as np
import mss
import time
from pynput.keyboard import Key, Controller
from pynput.mouse import Button, Controller as MouseController
from typing import Optional, Dict, Any

class GameFlowManager:
    def __init__(self):
        self.keyboard = Controller()
        self.mouse = MouseController()
        
        # Load templates for different screens
        self.templates: Dict[str, Optional[np.ndarray]] = {
            'death_screen': None,
            'skip_ad_button': None,
            'instructions_screen': None
        }
        
        # Load templates if they exist
        self.load_templates()
        
    def load_templates(self):
        """Load template images for screen detection"""
        try:
            death_template = cv2.imread("template/death_screen.png", cv2.IMREAD_GRAYSCALE)
            if death_template is not None:
                self.templates['death_screen'] = death_template
                print("Death screen template loaded")
            else:
                print("Death screen template not found - please create template/death_screen.png")
        except Exception as e:
            print(f"Error loading death screen template: {e}")
            
        try:
            skip_template = cv2.imread("template/skip_ad_button.png", cv2.IMREAD_GRAYSCALE)
            if skip_template is not None:
                self.templates['skip_ad_button'] = skip_template
                print("Skip ad button template loaded")
            else:
                print("Skip ad button template not found - please create template/skip_ad_button.png")
        except Exception as e:
            print(f"Error loading skip ad button template: {e}")
            
        try:
            instructions_template = cv2.imread("template/instructions_screen.png", cv2.IMREAD_GRAYSCALE)
            if instructions_template is not None:
                self.templates['instructions_screen'] = instructions_template
                print("Instructions screen template loaded")
            else:
                print("Instructions screen template not found - please create template/instructions_screen.png")
        except Exception as e:
            print(f"Error loading instructions screen template: {e}")
    
    def detect_screen(self, template_name, threshold=0.7):
        """Detect if a specific screen is visible"""
        template = self.templates.get(template_name)
        if template is None:
            return False, None
            
        with mss.mss() as sct:
            # Capture the game area (adjust ROI)
            screen = np.array(sct.grab(sct.monitors[0]))[:, :, :3]
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            
            # Template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            return max_val >= threshold, max_loc
    
    def handle_death_screen(self):
        """Handle the death screen by pressing space"""
        print("Detected death screen - pressing space to continue...")
        self.keyboard.press(Key.space)
        time.sleep(0.1)
        self.keyboard.release(Key.space)
        time.sleep(2)  # Wait for transition
    
    def handle_skip_ad(self):
        """Detect and click skip ad button"""
        print("Looking for skip ad button...")
        detected, location = self.detect_screen('skip_ad_button', threshold=0.6)
        
        if detected and location is not None:
            print("Skip ad button found - clicking...")
            # Click on the skip button
            self.mouse.position = (location[0] + 50, location[1] + 25) #adjust based button size
            self.mouse.click(Button.left)
            time.sleep(1)
            return True
        else:
            print("No skip ad button found - waiting...")
            return False
    
    def handle_instructions_screen(self):
        """Handle instructions screen by pressing space"""
        print("Detected instructions screen - pressing space to continue...")
        self.keyboard.press(Key.space)
        time.sleep(0.1)
        self.keyboard.release(Key.space)
        time.sleep(1)
    
    def wait_for_game_restart(self, max_wait_time=30):
        """Wait for the game to restart and return to normal gameplay"""
        print("Waiting for game to restart...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check for death screen
            death_detected, _ = self.detect_screen('death_screen')
            if death_detected:
                self.handle_death_screen()
                time.sleep(2)
                continue
            
            # Check for skip ad button
            skip_clicked = self.handle_skip_ad()
            if skip_clicked:
                time.sleep(2)
                continue
            
            # Check for instructions screen
            instructions_detected, _ = self.detect_screen('instructions_screen')
            if instructions_detected:
                self.handle_instructions_screen()
                time.sleep(2)
                continue
            
            # If none of the above, game might be ready
            print("Game appears to be ready")
            time.sleep(3)  # Extra wait to ensure game is fully loaded
            return True
            
        print("Timeout waiting for game restart")
        return False
    
    def is_game_ready(self):
        """Check if the game is in normal gameplay state"""
        # Check if any of the special screens are visible
        death_detected, _ = self.detect_screen('death_screen')
        skip_detected, _ = self.detect_screen('skip_ad_button')
        instructions_detected, _ = self.detect_screen('instructions_screen')
        
        # Game is ready if none of the special screens are visible
        return not (death_detected or skip_detected or instructions_detected) 