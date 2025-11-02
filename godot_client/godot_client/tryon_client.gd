extends Control

# UDP Socket for receiving webcam stream
var socket_udp := PacketPeerUDP.new()
var server_address := "127.0.0.1"
var server_port := 5005
var is_connected := false

# UDP Socket for sending commands to Python server
var command_socket := PacketPeerUDP.new()
var command_port := 5006  # Different port for commands

# Image processing
var image := Image.new()
var texture := ImageTexture.new()

# UI references
@onready var webcam_display: TextureRect = $WebcamContainer/WebcamDisplay
@onready var status_label: Label = $WebcamContainer/InfoOverlay/StatusPanel/VBox/StatusLabel
@onready var info_label: Label = $WebcamContainer/InfoOverlay/StatusPanel/VBox/InfoLabel
@onready var face_detection_label: Label = $WebcamContainer/InfoOverlay/StatusPanel/VBox/FaceDetectionLabel
@onready var mask_preview: TextureRect = $WebcamContainer/InfoOverlay/MaskPreviewPanel/MarginContainer/MaskPreview
@onready var settings_sidebar: PanelContainer = $SettingsSidebar
@onready var settings_controls: VBoxContainer = $SettingsSidebar/MarginContainer/VBoxContainer/ControlsContainer
@onready var mask_carousel: HBoxContainer = $WebcamContainer/MaskCarouselContainer/MarginContainer/VBox/MaskScrollContainer/CenterContainer/MaskHBoxContainer
@onready var toggle_menu_button: Button = $WebcamContainer/ToggleMenuButton
@onready var mask_carousel_container: PanelContainer = $WebcamContainer/MaskCarouselContainer
@onready var category_buttons: HBoxContainer = $WebcamContainer/MaskCarouselContainer/MarginContainer/VBox/CategoryButtonsContainer
@onready var mask_scroll: ScrollContainer = $WebcamContainer/MaskCarouselContainer/MarginContainer/VBox/MaskScrollContainer

# Mask categories
var mask_categories = {
	"medical": [1, 2, 3, 4, 5],
	"hazard": [6],
	"accessory": [7]
}

# Sidebar state
var settings_visible := false

# Menu state
var menu_visible := true

# Stats
var frame_count := 0
var fps_counter := 0
var fps_time := 0.0
var current_fps := 0.0

# Mask state (local tracking)
var current_mask_num := 1
var mask_enabled := false  # Start with mask OFF
var active_mask_buttons := {}  # Track buttons for highlighting

# Parameter sliders (to reset later)
var param_sliders := {}  # Store sliders by command_key for reset

# Utility: resolve mask path from common locations
func _get_mask_path(mask_num: int) -> String:
	var candidates = [
		"res://assets/mask%d.png" % mask_num,
		"res://../assets/mask%d.png" % mask_num,
		"res://../../assets/mask%d.png" % mask_num
	]
	for p in candidates:
		if FileAccess.file_exists(p):
			return p
	return ""

# Handler for slider value changes
func _on_slider_changed(value: float, command_key: String):
	# Called when slider value changes
	send_command("adjust_%s:%f" % [command_key, value])

func _add_param_row(container: Control, label_text: String, command_key: String, min_val: float, max_val: float, step: float, default_val: float):
	# Main VBox for this parameter
	var param_vbox = VBoxContainer.new()
	param_vbox.custom_minimum_size = Vector2(280, 70)
	
	# Label
	var lbl = Label.new()
	lbl.text = label_text
	lbl.add_theme_color_override("font_color", Color(1, 1, 1))
	lbl.add_theme_font_size_override("font_size", 14)
	param_vbox.add_child(lbl)
	
	# HBox for slider and buttons
	var controls_hbox = HBoxContainer.new()
	controls_hbox.custom_minimum_size = Vector2(280, 36)
	
	# Minus button
	var btn_minus = Button.new()
	btn_minus.text = "-"
	btn_minus.custom_minimum_size = Vector2(40, 36)
	controls_hbox.add_child(btn_minus)
	
	# Slider container with center line
	var slider_container = Control.new()
	slider_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider_container.custom_minimum_size = Vector2(140, 36)
	
	# Slider (add first so it's behind the line)
	var slider = HSlider.new()
	slider.min_value = min_val
	slider.max_value = max_val
	slider.step = step
	slider.value = default_val
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider.custom_minimum_size = Vector2(140, 36)
	slider.value_changed.connect(_on_slider_changed.bind(command_key))
	slider_container.add_child(slider)
	
	# Center line (visual indicator for zero/default) - drawn on top
	var center_line = ColorRect.new()
	center_line.color = Color(1, 0.85, 0.3, 0.7)  # Yellow-orange, more visible
	center_line.custom_minimum_size = Vector2(3, 28)
	center_line.mouse_filter = Control.MOUSE_FILTER_IGNORE  # Don't block slider interaction
	# Position will be updated when slider is resized
	slider.resized.connect(func():
		# Position at center of slider (accounting for margins)
		var center_x = slider.size.x / 2.0 - 1.5  # -1.5 to center the 3px line
		center_line.position = Vector2(center_x, 4)  # 4px from top for vertical centering
	)
	slider_container.add_child(center_line)
	
	controls_hbox.add_child(slider_container)
	
	# Store slider reference for reset
	param_sliders[command_key] = slider
	
	# Plus button
	var btn_plus = Button.new()
	btn_plus.text = "+"
	btn_plus.custom_minimum_size = Vector2(40, 36)
	controls_hbox.add_child(btn_plus)
	
	# Connect buttons to adjust slider value
	btn_minus.pressed.connect(func():
		var new_value = slider.value - step
		slider.value = clamp(new_value, slider.min_value, slider.max_value)
	)
	
	btn_plus.pressed.connect(func():
		var new_value = slider.value + step
		slider.value = clamp(new_value, slider.min_value, slider.max_value)
	)
	
	param_vbox.add_child(controls_hbox)
	container.add_child(param_vbox)

func _ready():
	print("Try-On Mask Godot Client Starting...")
	print("Server: %s:%d" % [server_address, server_port])
	
	# Bind to port to receive UDP packets
	var err = socket_udp.bind(server_port)
	if err != OK:
		print("Failed to bind to port %d: %s" % [server_port, error_string(err)])
		status_label.text = "Status: Failed to bind port!"
		status_label.modulate = Color(1, 0.3, 0.3)
		return
	
	print("Socket bound to port %d" % server_port)
	
	# Setup command socket (for sending commands to Python)
	# No need to bind, just use for sending
	print("Command socket ready for sending to Python server")
	
	is_connected = true
	status_label.text = "Status: Listening for frames..."
	status_label.modulate = Color(0.4, 1, 0.4)
	
	fps_time = Time.get_ticks_msec() / 1000.0
	
	print("\n=== CONTROLS ===")
	print("Q or ESC - Quit")
	print("M - Toggle Mask")
	print("1-7 - Switch Mask")
	print("S - Screenshot")
	print("================\n")
	
	# Ensure mask is OFF at startup
	await get_tree().create_timer(0.5).timeout  # Wait for connection
	send_command("mask_off")  # Custom command to turn off mask

	# Build overlay parameter controls
	var control_panel_title = Label.new()
	control_panel_title.text = "Overlay Parameters"
	control_panel_title.add_theme_color_override("font_color", Color(1, 1, 1))
	control_panel_title.add_theme_font_size_override("font_size", 16)
	control_panel_title.horizontal_alignment = 1
	settings_controls.add_child(control_panel_title)
	
	# Add spacing
	var spacer1 = Control.new()
	spacer1.custom_minimum_size = Vector2(0, 10)
	settings_controls.add_child(spacer1)
	
	# Add parameter rows with sliders
	# Parameters: (container, label, command_key, min, max, step, default)
	# All defaults set to 0.0 (middle of range)
	_add_param_row(settings_controls, "Scale Width", "scale_width", -0.5, 0.5, 0.05, 0.0)
	_add_param_row(settings_controls, "Scale Height", "scale_height", -0.5, 0.5, 0.05, 0.0)
	_add_param_row(settings_controls, "Y Offset", "y_offset", -0.3, 0.3, 0.02, 0.0)
	
	# Add spacing before reset button
	var spacer2 = Control.new()
	spacer2.custom_minimum_size = Vector2(0, 15)
	settings_controls.add_child(spacer2)
	
	# Add reset button
	var reset_button = Button.new()
	reset_button.text = "Reset to Default"
	reset_button.custom_minimum_size = Vector2(0, 36)
	reset_button.pressed.connect(_on_reset_parameters_pressed)
	settings_controls.add_child(reset_button)
	
	# Show medical masks by default in carousel
	show_mask_category("medical")

func _on_reset_parameters_pressed():
	"""Reset all parameter sliders to default (0.0)."""
	for command_key in param_sliders:
		var slider = param_sliders[command_key]
		slider.value = 0.0  # This will trigger value_changed signal
	print("Parameters reset to default")

func _on_toggle_settings_button_pressed():
	"""Toggle settings sidebar visibility."""
	settings_visible = !settings_visible
	settings_sidebar.visible = settings_visible

func _on_toggle_menu_button_pressed():
	"""Toggle mask menu visibility (category buttons + carousel)."""
	menu_visible = !menu_visible
	mask_carousel_container.visible = menu_visible
	
	# Update button text
	if menu_visible:
		toggle_menu_button.text = "▼ Sembunyikan Menu Masker"
	else:
		toggle_menu_button.text = "▲ Tampilkan Menu Masker"

func show_mask_category(category: String):
	"""Display mask buttons in horizontal carousel for selected category."""
	# Clear existing buttons
	for child in mask_carousel.get_children():
		child.queue_free()
	
	active_mask_buttons.clear()
	
	# Get mask numbers for this category
	var mask_numbers = mask_categories.get(category, [])
	
	# Create buttons for each mask in horizontal layout
	for mask_num in mask_numbers:
		# Create container for button with border (uniform style for all)
		var button_container = PanelContainer.new()
		button_container.custom_minimum_size = Vector2(120, 120)
		
		# Set default style - dark semi-transparent background with subtle border
		var style = StyleBoxFlat.new()
		style.bg_color = Color(0.1, 0.1, 0.1, 0.7)  # Dark gray semi-transparent (60% opacity)
		style.border_color = Color(0.4, 0.4, 0.4, 0.6)  # Subtle gray border
		style.set_border_width_all(2)
		style.corner_radius_top_left = 8
		style.corner_radius_top_right = 8
		style.corner_radius_bottom_left = 8
		style.corner_radius_bottom_right = 8
		button_container.add_theme_stylebox_override("panel", style)
		
		# Create button with image
		var button = TextureButton.new()
		button.custom_minimum_size = Vector2(120, 120)
		button.stretch_mode = TextureButton.STRETCH_KEEP_ASPECT_CENTERED
		
		# Try to load mask image
		var mask_path = _get_mask_path(mask_num)
		if mask_path != "" and FileAccess.file_exists(mask_path):
			var img = Image.new()
			var err = img.load(mask_path)
			if err == OK:
				# Resize image to fit button
				img.resize(120, 120, Image.INTERPOLATE_LANCZOS)
				button.texture_normal = ImageTexture.create_from_image(img)
		
		# If no image, use regular button with text
		if button.texture_normal == null:
			var regular_button = Button.new()
			regular_button.text = "Mask %d" % mask_num
			regular_button.custom_minimum_size = Vector2(120, 120)
			regular_button.pressed.connect(_on_mask_button_pressed.bind(mask_num))
			button_container.add_child(regular_button)
			mask_carousel.add_child(button_container)
			active_mask_buttons[mask_num] = button_container
		else:
			button.pressed.connect(_on_mask_button_pressed.bind(mask_num))
			button_container.add_child(button)
			mask_carousel.add_child(button_container)
			active_mask_buttons[mask_num] = button_container
	
	print("Showing %s masks in carousel: %s" % [category, mask_numbers])

func _on_medical_button_pressed():
	show_mask_category("medical")

func _on_hazard_button_pressed():
	show_mask_category("hazard")

func _on_accessory_button_pressed():
	show_mask_category("accessory")

func _on_mask_button_pressed(mask_num: int):
	"""Handle mask selection from carousel."""
	current_mask_num = mask_num
	print("Selected Mask %d from carousel" % mask_num)
	send_command("mask_%d" % mask_num)
	
	# Enable mask if it was off
	if not mask_enabled:
		mask_enabled = true
		# Mask will be enabled by the switch command automatically
	
	# Update highlight
	update_mask_highlight(mask_num)
	
	# Update mask preview
	update_mask_preview(mask_num)

func update_mask_highlight(selected_mask: int):
	"""Highlight the selected mask button with green border."""
	for mask_num in active_mask_buttons:
		var container = active_mask_buttons[mask_num]
		if container is PanelContainer:
			var style = StyleBoxFlat.new()
			# Dark semi-transparent background for all buttons
			style.bg_color = Color(0.1, 0.1, 0.1, 0.6)  # Dark gray semi-transparent (60% opacity)
			style.corner_radius_top_left = 8
			style.corner_radius_top_right = 8
			style.corner_radius_bottom_left = 8
			style.corner_radius_bottom_right = 8
			
			if mask_num == selected_mask:
				# Selected - green border (thick and bright)
				style.border_color = Color(0.2, 1.0, 0.2, 1.0)
				style.set_border_width_all(4)
			else:
				# Unselected - subtle gray border (normal)
				style.border_color = Color(0.4, 0.4, 0.4, 0.5)
				style.set_border_width_all(2)
			
			container.add_theme_stylebox_override("panel", style)

func update_mask_preview(mask_num: int):
	"""Update the mask preview image."""
	if mask_num > 0:
		# Try to load mask image
		var mask_path = _get_mask_path(mask_num)
		if mask_path != "" and FileAccess.file_exists(mask_path):
			var img = Image.new()
			var err = img.load(mask_path)
			if err == OK:
				mask_preview.texture = ImageTexture.create_from_image(img)
			else:
				mask_preview.texture = null
		else:
			mask_preview.texture = null
	else:
		# No mask active
		mask_preview.texture = null

func _process(_delta):
	if not is_connected:
		return
	
	# Handle input
	handle_input()
	
	# Check for incoming packets
	while socket_udp.get_available_packet_count() > 0:
		var packet = socket_udp.get_packet()
		process_packet(packet)

func handle_input():
	"""Handle keyboard input from Godot Input Map."""
	
	# Quit (Q or ESC)
	if Input.is_action_just_pressed("q"):
		print("Quit requested from Godot client")
		send_command("quit")
		await get_tree().create_timer(0.1).timeout  # Small delay to send command
		get_tree().quit()
	
	# Toggle Mask (M)
	if Input.is_action_just_pressed("m"):
		mask_enabled = not mask_enabled
		print("Toggle mask: %s" % ("ON" if mask_enabled else "OFF"))
		send_command("toggle_mask")
	
	# Screenshot (S)
	if Input.is_action_just_pressed("s"):
		print("Screenshot requested")
		send_command("screenshot")
	
	# Switch Mask (1-7)
	for i in range(1, 8):
		var action_name = str(i)
		if Input.is_action_just_pressed(action_name):
			current_mask_num = i
			print("Switch to mask%d" % i)
			send_command("mask_%d" % i)
			update_mask_preview(i)
			break

func send_command(command: String):
	"""Send command to Python server via UDP."""
	var packet = command.to_utf8_buffer()
	
	# Send to Python server (use same address but different handling)
	# Python needs to listen for commands
	command_socket.set_dest_address(server_address, command_port)
	var err = command_socket.put_packet(packet)
	
	if err != OK:
		print("Failed to send command '%s': %s" % [command, error_string(err)])
	else:
		print("Command sent: %s" % command)

func process_packet(packet: PackedByteArray):
	"""Process received UDP packet containing JPEG image and metadata."""
	
	if packet.size() < 6:  # 4 bytes frame_size + 1 byte num_faces + 1 byte mask_num
		print("Packet too small: %d bytes" % packet.size())
		return
	
	# Read frame size (first 4 bytes, big-endian)
	var frame_size = (packet[0] << 24) | (packet[1] << 16) | (packet[2] << 8) | packet[3]
	
	# Read num_faces (byte 4)
	var num_faces = packet[4]
	
	# Read mask_num (byte 5)
	var mask_num = packet[5]
	
	# Extract image data (skip first 6 bytes)
	var image_data = packet.slice(6)
	
	if image_data.size() != frame_size:
		print("Size mismatch: expected %d, got %d" % [frame_size, image_data.size()])
		return
	
	# Decode JPEG
	var err = image.load_jpg_from_buffer(image_data)
	if err != OK:
		print("Failed to decode JPEG: %s" % error_string(err))
		return
	
	# Update texture
	texture = ImageTexture.create_from_image(image)
	webcam_display.texture = texture
	
	# Update stats
	frame_count += 1
	fps_counter += 1
	
	var current_time = Time.get_ticks_msec() / 1000.0
	if current_time - fps_time >= 1.0:
		current_fps = fps_counter / (current_time - fps_time)
		fps_counter = 0
		fps_time = current_time
		
		# Update UI
		status_label.text = "Webcam: Aktif ✓"
		status_label.modulate = Color(0.3, 1, 0.3)
		info_label.text = "FPS: %.1f | %dx%d" % [
			current_fps,
			image.get_width(), 
			image.get_height()
		]
		
		# Update face detection label
		face_detection_label.text = "Wajah Terdeteksi: %d" % num_faces
		if num_faces > 0:
			face_detection_label.modulate = Color(0.3, 1.0, 0.3)
		else:
			face_detection_label.modulate = Color(0.8, 0.8, 0.8)
		
		# Update mask preview if mask changed
		if mask_num != current_mask_num:
			current_mask_num = mask_num
			update_mask_preview(mask_num)
			update_mask_highlight(mask_num)

func _exit_tree():
	socket_udp.close()
	command_socket.close()
	print("Sockets closed")
