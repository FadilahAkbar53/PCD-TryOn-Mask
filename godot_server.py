"""
Godot UDP Server - Send Try-On Mask results to Godot client.

Usage:
    python godot_server.py --camera 0 --host 127.0.0.1 --port 5005

Controls are same as webcam mode:
    'q' - Quit
    'm' - Toggle mask on/off
    '1-7' - Switch between mask1.png to mask7.png
"""
import socket
import struct
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from pipelines.infer import VideoInference
from pipelines.utils import logger


class GodotTryOnServer:
    """UDP Server that streams Try-On Mask results to Godot."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5005, 
                 model_dir: str = 'models', mask_dir: str = 'assets',
                 quality: int = 80, max_size: int = 1280):
        """
        Initialize Godot server.
        
        Args:
            host: Server host address
            port: Server port for streaming video
            model_dir: Directory containing trained models
            mask_dir: Directory containing mask files
            quality: JPEG compression quality (1-100)
            max_size: Maximum dimension for streaming (to reduce bandwidth)
        """
        self.host = host
        self.port = port
        self.quality = quality
        self.max_size = max_size
        
        # Initialize socket for video streaming
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        logger.info(f"UDP Socket initialized for streaming to {host}:{port}")
        
        # Initialize socket for receiving commands from Godot
        self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_port = 5006
        self.command_sock.bind(('0.0.0.0', self.command_port))
        self.command_sock.setblocking(False)  # Non-blocking to check without waiting
        logger.info(f"Command socket listening on port {self.command_port}")
        
        # Initialize inference pipeline
        self.inference = VideoInference(
            model_dir=model_dir,
            mask_dir=mask_dir
        )
        logger.info("Try-On Mask pipeline initialized")
        
        # Store user adjustments (delta from default)
        # These are applied to ALL masks when switching
        self.user_adjustments = {
            'scale_width': 0.0,
            'scale_height': 0.0,
            'y_offset': 0.0
        }
        
        # Client tracking
        self.client_address = None
        self.last_heartbeat = 0
        self.should_quit = False
        
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to max_size while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame
    
    def encode_frame(self, frame: np.ndarray) -> bytes:
        """Encode frame to JPEG bytes."""
        # Resize if needed
        frame = self.resize_frame(frame)
        
        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        return encoded.tobytes()
    
    def send_frame(self, frame_data: bytes, num_faces: int = 0, mask_num: int = 0):
        """Send frame data and metadata to Godot client via UDP."""
        if self.client_address is None:
            return
        
        # Prepare packet: [frame_size (4 bytes)] + [num_faces (1 byte)] + [mask_num (1 byte)] + [frame_data]
        frame_size = len(frame_data)
        packet = struct.pack('!I', frame_size) + struct.pack('!B', num_faces) + struct.pack('!B', mask_num) + frame_data
        
        # Send via UDP (may need chunking for large frames)
        try:
            self.sock.sendto(packet, self.client_address)
        except Exception as e:
            logger.warning(f"Failed to send frame: {e}")
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process frame with try-on mask and return result with metadata."""
        # Flip frame for detection
        frame_flipped = cv2.flip(frame, 1)
        
        # Detect faces
        detected_faces, scores = self.inference.detector.detect(frame_flipped)
        
        # Use face tracker for stability
        stable_faces = self.inference.face_tracker.update(detected_faces, scores)
        
        # Create result
        result = frame_flipped.copy()
        
        # Apply masks to stable faces if enabled
        if self.inference.mask_enabled and self.inference.current_mask and len(stable_faces) > 0:
            result = self.inference.current_mask.batch_overlay(result, stable_faces, enable_rotation=False)
        
        return result, len(stable_faces)
    
    def check_commands(self) -> str:
        """Check for incoming commands from Godot client."""
        try:
            data, addr = self.command_sock.recvfrom(1024)
            command = data.decode('utf-8').strip()
            logger.info(f"Received command from Godot: {command}")
            return command
        except BlockingIOError:
            # No data available
            return None
        except Exception as e:
            logger.warning(f"Error receiving command: {e}")
            return None
    
    def apply_user_adjustments(self):
        """Apply user adjustments to current mask."""
        if self.inference.current_mask:
            # Get mask's default/original values from _get_optimal_scaling()
            # This method returns (scale_width, scale_height, y_offset_ratio) based on mask ratio
            base_scale_w, base_scale_h, base_y_offset = self.inference.current_mask._get_optimal_scaling()
            
            # Apply user adjustments (additive from base values)
            self.inference.current_mask.scale_width = max(0.1, base_scale_w + self.user_adjustments['scale_width'])
            self.inference.current_mask.scale_height = max(0.1, base_scale_h + self.user_adjustments['scale_height'])
            self.inference.current_mask.y_offset_ratio = base_y_offset + self.user_adjustments['y_offset']
            
            logger.info(f"Applied user adjustments: scale_w={self.inference.current_mask.scale_width:.2f}, "
                       f"scale_h={self.inference.current_mask.scale_height:.2f}, "
                       f"y_offset={self.inference.current_mask.y_offset_ratio:.2f}")
    
    def handle_command(self, command: str, processed_frame: np.ndarray):
        """Handle command from Godot client."""
        if command == "quit":
            logger.info("Quit command received from Godot")
            self.should_quit = True
        
        elif command == "mask_off":
            # Force mask to be OFF
            if hasattr(self.inference, '_mask_enabled'):
                self.inference._mask_enabled = False
            logger.info("Mask disabled (startup)")
        
        elif command == "toggle_mask":
            self.inference.toggle_mask()
            logger.info(f"Mask {'enabled' if self.inference.mask_enabled else 'disabled'}")
        
        elif command.startswith("mask_"):
            try:
                mask_num = int(command.split("_")[1])
                if 1 <= mask_num <= 7:
                    self.inference.switch_mask(mask_num)
                    # Apply user adjustments to new mask
                    self.apply_user_adjustments()
                    logger.info(f"Switched to mask{mask_num}.png and applied user adjustments")
            except (ValueError, IndexError):
                logger.warning(f"Invalid mask command: {command}")
        
        elif command == "screenshot":
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            logger.info(f"Screenshot saved: {filename}")
        
        elif command.startswith("adjust_"):
            # Handle overlay parameter adjustments
            try:
                # Parse command: "adjust_scale_width:0.2"
                parts = command.split(":")
                if len(parts) == 2:
                    param_name = parts[0].replace("adjust_", "")
                    param_value = float(parts[1])  # Slider value from -0.5 to 0.5 (0 = default)
                    
                    # Store user adjustment (will be applied to all masks)
                    if param_name in ["scale_width", "scale_height", "y_offset"]:
                        self.user_adjustments[param_name] = param_value
                        
                        # Apply to current mask immediately
                        self.apply_user_adjustments()
                        
                        logger.info(f"User adjustment {param_name}: {param_value:+.2f}")
                    else:
                        logger.warning(f"Unknown parameter: {param_name}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Invalid adjust command: {command} - {e}")
        
        else:
            logger.warning(f"Unknown command: {command}")
    
    def run(self, camera_id: int = 0, show_preview: bool = False):
        """
        Run the server and stream webcam with try-on mask.
        
        Args:
            camera_id: Camera device ID
            show_preview: Show local preview window (default: False, use Godot for display)
        """
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        logger.info(f"Camera {camera_id} opened successfully")
        logger.info(f"Streaming to Godot client at {self.host}:{self.port}")
        logger.info(f"Listening for commands on port {self.command_port}")
        logger.info("Controls: Use Godot client UI or keyboard in this terminal")
        
        # Set client address (for simple UDP, we just use configured address)
        # In production, you'd wait for client heartbeat
        self.client_address = (self.host, self.port)
        
        fps_counter = 0
        fps_time = time.time()
        current_fps = 0
        frame_count = 0
        self.should_quit = False
        
        try:
            while True:
                # Check for commands from Godot
                command = self.check_commands()
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Process frame with try-on mask and get metadata
                processed_frame, num_faces = self.process_frame(frame)
                
                # Handle command if received (before encoding to save resources)
                if command:
                    self.handle_command(command, processed_frame)
                    if self.should_quit:
                        break
                
                # Get current mask number
                current_mask_num = self.inference.current_mask_id if self.inference.current_mask_id else 0
                
                # Encode and send to Godot with metadata
                frame_data = self.encode_frame(processed_frame)
                self.send_frame(frame_data, num_faces, current_mask_num)
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                
                # Show preview with FPS
                if show_preview:
                    preview = processed_frame.copy()
                    cv2.putText(preview, f"FPS: {current_fps}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(preview, f"Streaming to Godot {self.host}:{self.port}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow('Try-On Mask Server (Press Q to quit)', preview)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                else:
                    # No preview - minimal wait and print status to console
                    key = cv2.waitKey(1) & 0xFF
                    if fps_counter == 1:  # Print once per second
                        logger.info(f"Streaming... FPS: {current_fps} | Frames: {frame_count}")
                
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('m'):
                    self.inference.toggle_mask()
                    logger.info(f"Mask {'enabled' if self.inference.mask_enabled else 'disabled'}")
                elif ord('1') <= key <= ord('7'):
                    mask_num = key - ord('0')
                    self.inference.switch_mask(mask_num)
                    logger.info(f"Switched to mask{mask_num}.png")
                elif key == ord('s'):
                    # Screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    logger.info(f"Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.sock.close()
            self.command_sock.close()
            logger.info("Server stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Godot UDP Server for Try-On Mask Streaming'
    )
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Server host address')
    parser.add_argument('--port', type=int, default=5005,
                       help='Server UDP port')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory with trained models')
    parser.add_argument('--mask_dir', type=str, default='assets',
                       help='Directory containing mask files')
    parser.add_argument('--quality', type=int, default=80,
                       help='JPEG compression quality (1-100)')
    parser.add_argument('--max_size', type=int, default=1280,
                       help='Maximum frame dimension for streaming')
    parser.add_argument('--preview', action='store_true',
                       help='Enable local preview window (default: disabled, use Godot client)')
    
    args = parser.parse_args()
    
    # Create and run server
    server = GodotTryOnServer(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        mask_dir=args.mask_dir,
        quality=args.quality,
        max_size=args.max_size
    )
    
    server.run(
        camera_id=args.camera,
        show_preview=args.preview
    )


if __name__ == '__main__':
    main()
