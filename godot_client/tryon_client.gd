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
@onready var mask_preview: TextureRect = $WebcamContainer/InfoOverlay/MaskPreviewPanel/MarginContainer/VBoxContainer/ExpandableContent/MaskPreview
@onready var mask_title_button: Button = $WebcamContainer/InfoOverlay/MaskPreviewPanel/MarginContainer/VBoxContainer/MaskTitleButton
@onready var mask_description: RichTextLabel = $WebcamContainer/InfoOverlay/MaskPreviewPanel/MarginContainer/VBoxContainer/ExpandableContent/MaskDescription
@onready var expandable_content: VBoxContainer = $WebcamContainer/InfoOverlay/MaskPreviewPanel/MarginContainer/VBoxContainer/ExpandableContent
@onready var mask_preview_panel: PanelContainer = $WebcamContainer/InfoOverlay/MaskPreviewPanel
@onready var settings_sidebar: PanelContainer = $SettingsSidebar
@onready var settings_controls: VBoxContainer = $SettingsSidebar/MarginContainer/VBoxContainer/ScrollContainer/ControlsContainer
@onready var mask_carousel: HBoxContainer = $WebcamContainer/MaskCarouselContainer/MarginContainer/VBox/MaskScrollContainer/CenterContainer/MaskHBoxContainer
@onready var toggle_menu_button: Button = $WebcamContainer/ToggleMenuButton
@onready var mask_carousel_container: PanelContainer = $WebcamContainer/MaskCarouselContainer
@onready var category_buttons: HBoxContainer = $WebcamContainer/MaskCarouselContainer/MarginContainer/VBox/CategoryButtonsContainer
@onready var mask_scroll: ScrollContainer = $WebcamContainer/MaskCarouselContainer/MarginContainer/VBox/MaskScrollContainer
@onready var back_to_menu_button: Button = $BackToMenuButton

# Mask categories - loaded from JSON config
var mask_categories = {}
var mask_config = {}
var mask_names = {}  # Store mask names for display
var mask_descriptions = {}  # Store mask descriptions for display

# Sidebar state
var settings_visible := false

# Menu state
var menu_visible := true

# Mask preview panel state  
var preview_expanded := false

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

# Load mask configuration from JSON
func load_mask_config():
	# Try different paths
	var candidates = [
		"res://assets/mask_config.json",
		"res://../assets/mask_config.json",
		"res://../../assets/mask_config.json"
	]
	
	var config_file_path = ""
	for path in candidates:
		if FileAccess.file_exists(path):
			config_file_path = path
			break
	
	if config_file_path == "":
		print("Warning: mask_config.json not found, using fallback configuration")
		# Fallback to old configuration
		mask_categories = {
			"medical": [1, 2, 3, 4, 5],
			"utility": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
			"cultural": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
		}
		mask_names = {
			1: "Surgical", 2: "KF94", 3: "KN95", 4: "N95", 5: "N95 Plus",
			6: "Hazmat", 7: "Dust", 8: "Half Gas", 9: "M60G-1B", 10: "NP305",
			11: "P100", 12: "REikirc", 13: "Type 66", 14: "V400", 15: "V500",
			16: "Hannya", 17: "Panji", 18: "Samurai Menpo", 19: "Rangda", 20: "Noh",
			21: "Mexican", 22: "Kenala Merah", 23: "Dayak", 24: "Chinese Opera",
			25: "Barongsai", 26: "Barong"
		}
		mask_descriptions = {
			1: "Masker sekali pakai untuk tenaga medis",
			2: "Masker Korea dengan filtrasi tinggi",
			3: "Masker standar Tiongkok efisiensi 95%",
			4: "Respirator dengan filtrasi tinggi",
			5: "N95 dengan lapisan filtrasi ekstra",
			6: "Respirator untuk bahaya kimia",
			7: "Masker ringan untuk debu",
			8: "Masker setengah wajah dengan filter gas",
			9: "Model respirator militer",
			10: "Masker industri ringan",
			11: "Respirator dengan filtrasi 99.97%",
			12: "Masker respirator futuristik",
			13: "Masker minimalis area mulut",
			14: "Respirator modern dengan filter karbon",
			15: "Masker premium multi-layer",
			16: "Topeng tradisional Jepang",
			17: "Topeng Jawa dengan ekspresi tenang",
			18: "Masker samurai Jepang",
			19: "Topeng Bali ratu roh jahat",
			20: "Topeng teater Jepang klasik",
			21: "Topeng bergaya Meksiko",
			22: "Topeng Jawa berwarna merah",
			23: "Topeng suku Dayak Kalimantan",
			24: "Topeng tradisional Tiongkok",
			25: "Bagian wajah kostum Barongsai",
			26: "Topeng suci Bali"
		}
		return
	
	var file = FileAccess.open(config_file_path, FileAccess.READ)
	if file == null:
		print("Error: Could not open mask_config.json")
		return
	
	var json_string = file.get_as_text()
	file.close()
	
	var json = JSON.new()
	var parse_result = json.parse(json_string)
	
	if parse_result != OK:
		print("Error parsing mask_config.json: ", json.get_error_message())
		return
	
	mask_config = json.data
	
	# Build mask_categories and mask_names from config
	mask_categories.clear()
	mask_names.clear()
	mask_descriptions.clear()
	
	for category_key in mask_config:
		var category_data = mask_config[category_key]
		var mask_numbers = []
		
		for mask_id in category_data.masks:
			var mask_num = int(mask_id)
			mask_numbers.append(mask_num)
			mask_names[mask_num] = category_data.masks[mask_id].name
			mask_descriptions[mask_num] = category_data.masks[mask_id].get("description", "Deskripsi tidak tersedia")
		
		mask_categories[category_key] = mask_numbers
	
	print("Loaded mask configuration:")
	for category in mask_categories:
		print("  %s: %s" % [category, mask_categories[category]])

# Utility: resolve mask path from common locations (updated with filename)
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

func _create_unified_button_style(state: String) -> StyleBoxFlat:
	"""Create unified button style for all control buttons (Back, Settings, Toggle Menu).
	State: 'normal', 'hover', or 'pressed'"""
	var style = StyleBoxFlat.new()
	
	match state:
		"normal":
			style.bg_color = Color(0.15, 0.2, 0.3, 0.85)  # Consistent dark blue
			style.border_color = Color(0.3, 0.4, 0.5, 0.7)
			style.shadow_size = 4
			style.shadow_color = Color(0, 0, 0, 0.4)
		"hover":
			style.bg_color = Color(0.2, 0.3, 0.45, 0.95)  # Brighter blue
			style.border_color = Color(0.4, 0.55, 0.7, 0.9)
			style.shadow_size = 6
			style.shadow_color = Color(0, 0, 0, 0.5)
		"pressed":
			style.bg_color = Color(0.1, 0.15, 0.25, 0.9)  # Darker blue
			style.border_color = Color(0.25, 0.35, 0.45, 0.8)
			style.shadow_size = 2
			style.shadow_color = Color(0, 0, 0, 0.3)
	
	style.set_border_width_all(2)
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.shadow_offset = Vector2(0, 2)
	
	return style

func setup_back_button():
	"""Apply unified button style to Back to Menu button."""
	back_to_menu_button.add_theme_stylebox_override("normal", _create_unified_button_style("normal"))
	back_to_menu_button.add_theme_stylebox_override("hover", _create_unified_button_style("hover"))
	back_to_menu_button.add_theme_stylebox_override("pressed", _create_unified_button_style("pressed"))
	back_to_menu_button.add_theme_color_override("font_color", Color(0.9, 0.95, 1, 1))

func _on_back_to_menu_pressed():
	"""Return to main menu and cleanup resources."""
	print("Returning to main menu...")
	# Send disconnect command (NOT quit - server should keep running)
	send_command("disconnect")
	# Give time for command to send
	await get_tree().create_timer(0.1).timeout
	# Close sockets
	socket_udp.close()
	command_socket.close()
	# Change to main menu scene
	get_tree().change_scene_to_file("res://main_menu.tscn")

# Handler for slider value changes
func _on_slider_changed(value: float, command_key: String, value_label: Label):
	# Called when slider value changes
	send_command("adjust_%s:%f" % [command_key, value])
	# Update value display with color coding
	value_label.text = "%.2f" % value
	
	# Color coding based on value
	if abs(value) < 0.01:  # Near zero (default)
		value_label.add_theme_color_override("font_color", Color(0.8, 1, 0.8))  # Light green
	elif value < 0:  # Negative values
		value_label.add_theme_color_override("font_color", Color(1, 0.8, 0.8))  # Light red
	else:  # Positive values
		value_label.add_theme_color_override("font_color", Color(0.8, 0.8, 1))  # Light blue

func _add_param_row(container: Control, label_text: String, command_key: String, min_val: float, max_val: float, step: float, default_val: float):
	# Main VBox for this parameter (more compact)
	var param_vbox = VBoxContainer.new()
	param_vbox.custom_minimum_size = Vector2(260, 65)
	
	# Header HBox for label and value
	var header_hbox = HBoxContainer.new()
	header_hbox.custom_minimum_size = Vector2(260, 18)
	
	# Label
	var lbl = Label.new()
	lbl.text = label_text
	lbl.add_theme_color_override("font_color", Color(0.95, 0.95, 1))
	lbl.add_theme_font_size_override("font_size", 12)
	lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header_hbox.add_child(lbl)
	
	# Value display label
	var value_label = Label.new()
	value_label.text = "%.2f" % default_val
	value_label.add_theme_color_override("font_color", Color(0.8, 1, 0.8))  # Light green
	value_label.add_theme_font_size_override("font_size", 11)
	value_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	value_label.custom_minimum_size = Vector2(50, 18)
	header_hbox.add_child(value_label)
	
	# Add range indicator (smaller)
	var range_label = Label.new()
	range_label.text = "[%.1f, %.1f]" % [min_val, max_val]
	range_label.add_theme_color_override("font_color", Color(0.6, 0.6, 0.65))
	range_label.add_theme_font_size_override("font_size", 9)
	range_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	range_label.custom_minimum_size = Vector2(70, 18)
	header_hbox.add_child(range_label)
	
	param_vbox.add_child(header_hbox)
	
	# HBox for slider and buttons (more compact)
	var controls_hbox = HBoxContainer.new()
	controls_hbox.custom_minimum_size = Vector2(260, 32)
	
	# Minus button (smaller)
	var btn_minus = Button.new()
	btn_minus.text = "-"
	btn_minus.custom_minimum_size = Vector2(32, 28)
	btn_minus.add_theme_font_size_override("font_size", 16)
	controls_hbox.add_child(btn_minus)
	
	# Slider container with center line
	var slider_container = Control.new()
	slider_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider_container.custom_minimum_size = Vector2(140, 32)
	
	# Slider (add first so it's behind the line)
	var slider = HSlider.new()
	slider.min_value = min_val
	slider.max_value = max_val
	slider.step = step
	slider.value = default_val
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider.custom_minimum_size = Vector2(140, 28)
	slider.value_changed.connect(_on_slider_changed.bind(command_key, value_label))
	slider_container.add_child(slider)
	
	# Center line (visual indicator for zero/default) - drawn on top
	var center_line = ColorRect.new()
	center_line.color = Color(1, 0.85, 0.3, 0.7)  # Yellow-orange
	center_line.custom_minimum_size = Vector2(2, 24)
	center_line.mouse_filter = Control.MOUSE_FILTER_IGNORE
	# Position will be updated when slider is resized
	slider.resized.connect(func():
		# Position at center of slider (accounting for margins)
		var center_x = slider.size.x / 2.0 - 1.0  # -1.0 to center the 2px line
		center_line.position = Vector2(center_x, 2)  # 2px from top for vertical centering
	)
	slider_container.add_child(center_line)
	
	controls_hbox.add_child(slider_container)
	
	# Store slider reference for reset
	param_sliders[command_key] = slider
	
	# Plus button (smaller)
	var btn_plus = Button.new()
	btn_plus.text = "+"
	btn_plus.custom_minimum_size = Vector2(32, 28)
	btn_plus.add_theme_font_size_override("font_size", 16)
	controls_hbox.add_child(btn_plus)
	
	# Connect buttons to adjust slider value (they will trigger value_changed signal)
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
	
	# Load mask configuration
	load_mask_config()
	
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
	
	# Connect mask title button signal
	mask_title_button.pressed.connect(_on_mask_title_button_pressed)
	
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
	
	# Setup Back to Menu button styling
	setup_back_button()

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
	_add_param_row(settings_controls, "X Offset", "x_offset", -0.3, 0.3, 0.02, 0.0)
	_add_param_row(settings_controls, "Transparency", "transparency", 0.0, 1.0, 0.05, 1.0)  # 0=invisible, 1=opaque
	
	# Add spacing for rotation section
	var spacer_rotation = Control.new()
	spacer_rotation.custom_minimum_size = Vector2(0, 10)
	settings_controls.add_child(spacer_rotation)
	
	# Add rotation section title
	var rotation_title = Label.new()
	rotation_title.text = "3D Rotation Control"
	rotation_title.add_theme_color_override("font_color", Color(1, 1, 0.8))
	rotation_title.add_theme_font_size_override("font_size", 14)
	rotation_title.horizontal_alignment = 1
	settings_controls.add_child(rotation_title)
	
	# Add 3D rotation toggle
	var rotation_toggle_hbox = HBoxContainer.new()
	rotation_toggle_hbox.custom_minimum_size = Vector2(280, 30)
	
	var rotation_label = Label.new()
	rotation_label.text = "Enable 3D Rotation"
	rotation_label.add_theme_color_override("font_color", Color(1, 1, 1))
	rotation_label.add_theme_font_size_override("font_size", 13)
	rotation_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	rotation_toggle_hbox.add_child(rotation_label)
	
	var rotation_checkbox = CheckBox.new()
	rotation_checkbox.text = ""
	rotation_checkbox.button_pressed = true  # Enable 3D rotation by default
	rotation_checkbox.custom_minimum_size = Vector2(30, 30)
	rotation_checkbox.toggled.connect(_on_3d_rotation_toggled)
	rotation_toggle_hbox.add_child(rotation_checkbox)
	
	settings_controls.add_child(rotation_toggle_hbox)
	
	# Add spacing before reset button
	var spacer2 = Control.new()
	spacer2.custom_minimum_size = Vector2(0, 10)
	settings_controls.add_child(spacer2)
	
	# Add reset button with modern styling
	var reset_button = Button.new()
	reset_button.text = "🔄 Reset to Default"
	reset_button.custom_minimum_size = Vector2(0, 32)
	
	# Style reset button
	var reset_normal = StyleBoxFlat.new()
	reset_normal.bg_color = Color(0.2, 0.25, 0.35, 0.9)
	reset_normal.set_border_width_all(2)
	reset_normal.border_color = Color(0.35, 0.45, 0.6, 0.8)
	reset_normal.corner_radius_top_left = 6
	reset_normal.corner_radius_top_right = 6
	reset_normal.corner_radius_bottom_left = 6
	reset_normal.corner_radius_bottom_right = 6
	
	var reset_hover = StyleBoxFlat.new()
	reset_hover.bg_color = Color(0.25, 0.35, 0.5, 0.95)
	reset_hover.set_border_width_all(2)
	reset_hover.border_color = Color(0.45, 0.6, 0.8, 0.9)
	reset_hover.corner_radius_top_left = 6
	reset_hover.corner_radius_top_right = 6
	reset_hover.corner_radius_bottom_left = 6
	reset_hover.corner_radius_bottom_right = 6
	
	var reset_pressed = StyleBoxFlat.new()
	reset_pressed.bg_color = Color(0.15, 0.2, 0.3, 0.9)
	reset_pressed.set_border_width_all(2)
	reset_pressed.border_color = Color(0.3, 0.4, 0.5, 0.8)
	reset_pressed.corner_radius_top_left = 6
	reset_pressed.corner_radius_top_right = 6
	reset_pressed.corner_radius_bottom_left = 6
	reset_pressed.corner_radius_bottom_right = 6
	
	reset_button.add_theme_stylebox_override("normal", reset_normal)
	reset_button.add_theme_stylebox_override("hover", reset_hover)
	reset_button.add_theme_stylebox_override("pressed", reset_pressed)
	reset_button.add_theme_color_override("font_color", Color(0.9, 0.95, 1, 1))
	reset_button.add_theme_font_size_override("font_size", 13)
	
	reset_button.pressed.connect(_on_reset_parameters_pressed)
	settings_controls.add_child(reset_button)
	
	# Show medical masks by default in carousel
	show_mask_category("medical")
	
	# Initialize mask preview (collapsed by default)
	preview_expanded = false
	expandable_content.visible = false
	mask_preview_panel.custom_minimum_size.y = 50
	
	# Set initial mask preview to "no mask selected" state
	mask_title_button.text = "▶ Pilih Masker"
	
	# Initialize menu state - START WITH MENU VISIBLE
	menu_visible = true
	mask_carousel_container.visible = true
	toggle_menu_button.text = "▼ Sembunyikan Menu Masker"
	
	# Setup category button styles and hover effects
	setup_category_buttons()
	
	# Setup toggle menu button style
	setup_toggle_menu_button()
	
	# Setup scroll container style
	setup_scroll_container()
	
	# Setup settings sidebar and buttons
	setup_settings_sidebar()
	setup_settings_buttons()

func setup_scroll_container():
	"""Setup modern scrollbar style for mask carousel."""
	# Get the ScrollContainer
	var scroll = mask_scroll as ScrollContainer
	
	# Create modern scrollbar style - horizontal only
	var scrollbar_style = StyleBoxFlat.new()
	scrollbar_style.bg_color = Color(0.2, 0.2, 0.25, 0.6)  # Dark semi-transparent
	scrollbar_style.corner_radius_top_left = 4
	scrollbar_style.corner_radius_top_right = 4
	scrollbar_style.corner_radius_bottom_left = 4
	scrollbar_style.corner_radius_bottom_right = 4
	
	var grabber_style = StyleBoxFlat.new()
	grabber_style.bg_color = Color(0.4, 0.5, 0.7, 0.8)  # Blue-ish grabber
	grabber_style.corner_radius_top_left = 4
	grabber_style.corner_radius_top_right = 4
	grabber_style.corner_radius_bottom_left = 4
	grabber_style.corner_radius_bottom_right = 4
	
	var grabber_hover_style = StyleBoxFlat.new()
	grabber_hover_style.bg_color = Color(0.5, 0.65, 0.9, 0.95)  # Brighter on hover
	grabber_hover_style.corner_radius_top_left = 4
	grabber_hover_style.corner_radius_top_right = 4
	grabber_hover_style.corner_radius_bottom_left = 4
	grabber_hover_style.corner_radius_bottom_right = 4
	
	# Apply to horizontal scrollbar
	scroll.add_theme_stylebox_override("scroll", scrollbar_style)
	scroll.add_theme_stylebox_override("scroll_focus", scrollbar_style)
	scroll.add_theme_stylebox_override("grabber", grabber_style)
	scroll.add_theme_stylebox_override("grabber_highlight", grabber_hover_style)
	scroll.add_theme_stylebox_override("grabber_pressed", grabber_hover_style)

func setup_settings_sidebar():
	"""Setup modern theme for settings sidebar panel."""
	# Create modern panel background style - MORE TRANSPARENT
	var panel_style = StyleBoxFlat.new()
	panel_style.bg_color = Color(0.12, 0.15, 0.2, 0.75)  # More transparent (0.75 instead of 0.95)
	panel_style.set_border_width_all(2)
	panel_style.border_color = Color(0.25, 0.3, 0.4, 0.6)  # Lighter border
	panel_style.corner_radius_top_left = 8
	panel_style.corner_radius_top_right = 8
	panel_style.corner_radius_bottom_left = 8
	panel_style.corner_radius_bottom_right = 8
	panel_style.shadow_size = 8
	panel_style.shadow_color = Color(0, 0, 0, 0.4)  # Softer shadow
	
	settings_sidebar.add_theme_stylebox_override("panel", panel_style)
	
	# Style the scrollbar in settings
	var settings_scroll = settings_sidebar.get_node("MarginContainer/VBoxContainer/ScrollContainer")
	
	var scrollbar_style = StyleBoxFlat.new()
	scrollbar_style.bg_color = Color(0.15, 0.18, 0.25, 0.6)
	scrollbar_style.corner_radius_top_left = 4
	scrollbar_style.corner_radius_top_right = 4
	scrollbar_style.corner_radius_bottom_left = 4
	scrollbar_style.corner_radius_bottom_right = 4
	
	var grabber_style = StyleBoxFlat.new()
	grabber_style.bg_color = Color(0.35, 0.45, 0.6, 0.8)
	grabber_style.corner_radius_top_left = 4
	grabber_style.corner_radius_top_right = 4
	grabber_style.corner_radius_bottom_left = 4
	grabber_style.corner_radius_bottom_right = 4
	
	var grabber_hover_style = StyleBoxFlat.new()
	grabber_hover_style.bg_color = Color(0.45, 0.6, 0.8, 0.95)
	grabber_hover_style.corner_radius_top_left = 4
	grabber_hover_style.corner_radius_top_right = 4
	grabber_hover_style.corner_radius_bottom_left = 4
	grabber_hover_style.corner_radius_bottom_right = 4
	
	settings_scroll.add_theme_stylebox_override("scroll", scrollbar_style)
	settings_scroll.add_theme_stylebox_override("scroll_focus", scrollbar_style)
	settings_scroll.add_theme_stylebox_override("grabber", grabber_style)
	settings_scroll.add_theme_stylebox_override("grabber_highlight", grabber_hover_style)
	settings_scroll.add_theme_stylebox_override("grabber_pressed", grabber_hover_style)

func setup_settings_buttons():
	"""Setup modern theme for settings toggle button and reset button."""
	# Get the toggle settings button
	var toggle_settings_btn = $ToggleSettingsButton
	
	# Use unified button style (same as Back button and Toggle Menu)
	toggle_settings_btn.add_theme_stylebox_override("normal", _create_unified_button_style("normal"))
	toggle_settings_btn.add_theme_stylebox_override("hover", _create_unified_button_style("hover"))
	toggle_settings_btn.add_theme_stylebox_override("pressed", _create_unified_button_style("pressed"))
	toggle_settings_btn.add_theme_color_override("font_color", Color(0.9, 0.95, 1, 1))

func setup_toggle_menu_button():
	"""Setup modern style for toggle menu button."""
	# Use unified button style (same as all control buttons)
	toggle_menu_button.add_theme_stylebox_override("normal", _create_unified_button_style("normal"))
	toggle_menu_button.add_theme_stylebox_override("hover", _create_unified_button_style("hover"))
	toggle_menu_button.add_theme_stylebox_override("pressed", _create_unified_button_style("pressed"))
	toggle_menu_button.add_theme_color_override("font_color", Color(0.9, 0.95, 1.0))
	toggle_menu_button.add_theme_color_override("font_hover_color", Color(1.0, 1.0, 1.0))

func setup_category_buttons():
	"""Setup visual styles and hover effects for category buttons."""
	var medical_btn = category_buttons.get_node("MedicalButton")
	var utility_btn = category_buttons.get_node("UtilityButton")
	var cultural_btn = category_buttons.get_node("CulturalButton")
	
	# Medical button - Green theme (changed from blue)
	var medical_style = StyleBoxFlat.new()
	medical_style.bg_color = Color(0.15, 0.35, 0.2, 0.8)  # Green
	medical_style.border_color = Color(0.25, 0.55, 0.35, 0.6)  # Green border
	medical_style.set_border_width_all(2)
	medical_style.corner_radius_top_left = 8
	medical_style.corner_radius_top_right = 8
	medical_style.corner_radius_bottom_left = 8
	medical_style.corner_radius_bottom_right = 8
	medical_btn.add_theme_stylebox_override("normal", medical_style)
	
	# Utility button - Orange theme
	var utility_style = StyleBoxFlat.new()
	utility_style.bg_color = Color(0.35, 0.2, 0.1, 0.8)
	utility_style.border_color = Color(0.6, 0.4, 0.2, 0.6)
	utility_style.set_border_width_all(2)
	utility_style.corner_radius_top_left = 8
	utility_style.corner_radius_top_right = 8
	utility_style.corner_radius_bottom_left = 8
	utility_style.corner_radius_bottom_right = 8
	utility_btn.add_theme_stylebox_override("normal", utility_style)
	
	# Cultural button - Purple theme
	var cultural_style = StyleBoxFlat.new()
	cultural_style.bg_color = Color(0.25, 0.15, 0.35, 0.8)
	cultural_style.border_color = Color(0.5, 0.3, 0.6, 0.6)
	cultural_style.set_border_width_all(2)
	cultural_style.corner_radius_top_left = 8
	cultural_style.corner_radius_top_right = 8
	cultural_style.corner_radius_bottom_left = 8
	cultural_style.corner_radius_bottom_right = 8
	cultural_btn.add_theme_stylebox_override("normal", cultural_style)
	
	# Add hover effects
	medical_btn.mouse_entered.connect(_on_category_button_hover.bind(medical_btn, "medical"))
	medical_btn.mouse_exited.connect(_on_category_button_unhover.bind(medical_btn, "medical"))
	
	utility_btn.mouse_entered.connect(_on_category_button_hover.bind(utility_btn, "utility"))
	utility_btn.mouse_exited.connect(_on_category_button_unhover.bind(utility_btn, "utility"))
	
	cultural_btn.mouse_entered.connect(_on_category_button_hover.bind(cultural_btn, "cultural"))
	cultural_btn.mouse_exited.connect(_on_category_button_unhover.bind(cultural_btn, "cultural"))

func _on_category_button_hover(button: Button, category: String):
	"""Animate category button on hover."""
	var hover_style = StyleBoxFlat.new()
	
	match category:
		"medical":
			hover_style.bg_color = Color(0.2, 0.5, 0.3, 0.95)  # Brighter green
			hover_style.border_color = Color(0.35, 0.75, 0.5, 0.9)  # Bright green border
		"utility":
			hover_style.bg_color = Color(0.5, 0.3, 0.15, 0.95)
			hover_style.border_color = Color(0.8, 0.55, 0.3, 0.9)
		"cultural":
			hover_style.bg_color = Color(0.35, 0.2, 0.5, 0.95)
			hover_style.border_color = Color(0.65, 0.4, 0.8, 0.9)
	
	hover_style.set_border_width_all(3)
	hover_style.corner_radius_top_left = 8
	hover_style.corner_radius_top_right = 8
	hover_style.corner_radius_bottom_left = 8
	hover_style.corner_radius_bottom_right = 8
	hover_style.shadow_color = Color(0, 0, 0, 0.4)
	hover_style.shadow_size = 6
	hover_style.shadow_offset = Vector2(0, 3)
	
	button.add_theme_stylebox_override("hover", hover_style)
	
	# Scale animation
	var tween = create_tween()
	tween.set_ease(Tween.EASE_OUT)
	tween.set_trans(Tween.TRANS_BACK)
	tween.tween_property(button, "scale", Vector2(1.05, 1.05), 0.2)

func _on_category_button_unhover(button: Button, category: String):
	"""Reset category button on unhover."""
	var tween = create_tween()
	tween.set_ease(Tween.EASE_OUT)
	tween.set_trans(Tween.TRANS_BACK)
	tween.tween_property(button, "scale", Vector2(1.0, 1.0), 0.2)

func _on_reset_parameters_pressed():
	"""Reset all parameter sliders to default values."""
	for command_key in param_sliders:
		var slider = param_sliders[command_key]
		if command_key == "transparency":
			slider.value = 1.0  # Transparency default is 1.0 (opaque)
		else:
			slider.value = 0.0  # Other parameters default to 0.0
		# This will trigger value_changed signal
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
	
	# Set background color based on category (matching main menu theme)
	var bg_color_rect = mask_carousel_container.get_node("BgStyle") as ColorRect
	match category:
		"medical":
			# Medical: Green theme (medical, natural, health)
			bg_color_rect.color = Color(0.12, 0.25, 0.15, 0.85)  # Deep green
		"utility":
			# Utility: Orange-brown (industrial, safety)
			bg_color_rect.color = Color(0.25, 0.15, 0.1, 0.85)  # Dark orange-brown
		"cultural":
			# Cultural: Purple-magenta (artistic, traditional)
			bg_color_rect.color = Color(0.2, 0.1, 0.25, 0.85)  # Deep purple
		_:
			bg_color_rect.color = Color(0, 0, 0, 0.5)  # Default dark
	
	# Get mask numbers for this category
	var mask_numbers = mask_categories.get(category, [])
	
	# Create buttons for each mask in horizontal layout
	for mask_num in mask_numbers:
		# Create container for button with border
		var button_container = PanelContainer.new()
		button_container.custom_minimum_size = Vector2(140, 150)
		
		# Set default style with category-specific accent
		var style = StyleBoxFlat.new()
		
		# Base style - semi-transparent with subtle gradient
		match category:
			"medical":
				style.bg_color = Color(0.15, 0.3, 0.2, 0.3)  # Light green tint
				style.border_color = Color(0.3, 0.6, 0.4, 0.6)  # Green border
			"utility":
				style.bg_color = Color(0.3, 0.2, 0.15, 0.3)  # Orange tint
				style.border_color = Color(0.8, 0.5, 0.3, 0.6)  # Orange border
			"cultural":
				style.bg_color = Color(0.25, 0.15, 0.3, 0.3)  # Purple tint
				style.border_color = Color(0.7, 0.4, 0.8, 0.6)  # Purple border
			_:
				style.bg_color = Color(1.0, 1.0, 1.0, 0.15)
				style.border_color = Color(1.0, 1.0, 1.0, 0.8)
		
		style.set_border_width_all(2)
		style.corner_radius_top_left = 10
		style.corner_radius_top_right = 10
		style.corner_radius_bottom_left = 10
		style.corner_radius_bottom_right = 10
		
		# Add subtle shadow
		style.shadow_color = Color(0, 0, 0, 0.3)
		style.shadow_size = 4
		style.shadow_offset = Vector2(0, 2)
		
		button_container.add_theme_stylebox_override("panel", style)
		
		# Store original style for animation
		button_container.set_meta("original_style", style)
		button_container.set_meta("mask_num", mask_num)
		button_container.set_meta("category", category)
		
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
			var mask_display_name = mask_names.get(mask_num, "Mask %d" % mask_num)
			regular_button.text = mask_display_name
			regular_button.custom_minimum_size = Vector2(120, 120)
			regular_button.pressed.connect(_on_mask_button_pressed.bind(mask_num))
			
			# Add hover effects for regular button
			regular_button.mouse_entered.connect(_on_mask_item_hover.bind(button_container))
			regular_button.mouse_exited.connect(_on_mask_item_unhover.bind(button_container))
			
			button_container.add_child(regular_button)
			mask_carousel.add_child(button_container)
			active_mask_buttons[mask_num] = button_container
		else:
			# Add label below texture button for mask name
			var vbox_with_label = VBoxContainer.new()
			vbox_with_label.custom_minimum_size = Vector2(120, 150)
			
			button.pressed.connect(_on_mask_button_pressed.bind(mask_num))
			
			# Add hover effects for texture button
			button.mouse_entered.connect(_on_mask_item_hover.bind(button_container))
			button.mouse_exited.connect(_on_mask_item_unhover.bind(button_container))
			
			vbox_with_label.add_child(button)
			
			# Add mask name label
			var name_label = Label.new()
			var mask_display_name = mask_names.get(mask_num, "Mask %d" % mask_num)
			name_label.text = mask_display_name
			name_label.add_theme_color_override("font_color", Color(1, 1, 1))
			name_label.add_theme_font_size_override("font_size", 12)
			name_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
			name_label.custom_minimum_size = Vector2(120, 30)
			name_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
			vbox_with_label.add_child(name_label)
			
			button_container.add_child(vbox_with_label)
			mask_carousel.add_child(button_container)
			active_mask_buttons[mask_num] = button_container
	
	print("Showing %s masks in carousel: %s" % [category, mask_numbers])

func _on_medical_button_pressed():
	show_mask_category("medical")

func _on_utility_button_pressed():
	show_mask_category("utility")

func _on_cultural_button_pressed():
	show_mask_category("cultural")

func _on_mask_item_hover(container: PanelContainer):
	"""Animate mask item on hover - scale up and brighten."""
	var category = container.get_meta("category", "")
	
	# Create hover style with brighter colors
	var hover_style = StyleBoxFlat.new()
	
	match category:
		"medical":
			hover_style.bg_color = Color(0.25, 0.5, 0.35, 0.6)  # Brighter green
			hover_style.border_color = Color(0.4, 0.8, 0.55, 0.9)  # Bright green border
		"utility":
			hover_style.bg_color = Color(0.45, 0.3, 0.2, 0.6)  # Brighter orange
			hover_style.border_color = Color(1.0, 0.65, 0.4, 0.9)  # Bright orange border
		"cultural":
			hover_style.bg_color = Color(0.35, 0.25, 0.45, 0.6)  # Brighter purple
			hover_style.border_color = Color(0.9, 0.6, 1.0, 0.9)  # Bright purple border
		_:
			hover_style.bg_color = Color(1.0, 1.0, 1.0, 0.3)
			hover_style.border_color = Color(1.0, 1.0, 1.0, 1.0)
	
	hover_style.set_border_width_all(3)  # Thicker border
	hover_style.corner_radius_top_left = 10
	hover_style.corner_radius_top_right = 10
	hover_style.corner_radius_bottom_left = 10
	hover_style.corner_radius_bottom_right = 10
	
	# Enhanced shadow on hover
	hover_style.shadow_color = Color(0, 0, 0, 0.5)
	hover_style.shadow_size = 8
	hover_style.shadow_offset = Vector2(0, 4)
	
	# Animate style change and scale
	var tween = create_tween()
	tween.set_parallel(true)
	tween.set_ease(Tween.EASE_OUT)
	tween.set_trans(Tween.TRANS_BACK)
	
	# Scale up
	tween.tween_property(container, "scale", Vector2(1.1, 1.1), 0.3)
	
	# Update style (immediate for visual feedback)
	container.add_theme_stylebox_override("panel", hover_style)

func _on_mask_item_unhover(container: PanelContainer):
	"""Reset mask item animation on unhover - but check if it's selected first."""
	var mask_num = container.get_meta("mask_num", 0)
	
	# If this is the currently selected mask, don't reset - keep it highlighted
	if mask_num == current_mask_num:
		# Keep selected style (same as hover) but reset scale
		var tween = create_tween()
		tween.set_ease(Tween.EASE_OUT)
		tween.set_trans(Tween.TRANS_BACK)
		tween.tween_property(container, "scale", Vector2(1.0, 1.0), 0.3)
		return
	
	# Get original style for non-selected items
	var original_style = container.get_meta("original_style")
	
	# Animate back to normal
	var tween = create_tween()
	tween.set_parallel(true)
	tween.set_ease(Tween.EASE_OUT)
	tween.set_trans(Tween.TRANS_BACK)
	
	# Scale back to normal
	tween.tween_property(container, "scale", Vector2(1.0, 1.0), 0.3)
	
	# Restore original style (immediate)
	container.add_theme_stylebox_override("panel", original_style)

func _on_mask_button_pressed(mask_num: int):
	"""Handle mask selection from carousel."""
	current_mask_num = mask_num
	print("Selected Mask %d from carousel" % mask_num)
	send_command("mask_%d" % mask_num)
	
	# Enable mask if it was off
	if not mask_enabled:
		mask_enabled = true
		send_command("toggle_mask")
	
	# Update highlight
	update_mask_highlight(mask_num)
	
	# FIXED: Always update mask preview (even if collapsed)
	# This ensures the title is updated and data is ready when user clicks info
	update_mask_preview(mask_num)

func update_mask_highlight(selected_mask: int):
	"""Highlight the selected mask button with active style (same as hover)."""
	for mask_num in active_mask_buttons:
		var container = active_mask_buttons[mask_num]
		if container is PanelContainer:
			var category = container.get_meta("category", "")
			var style = StyleBoxFlat.new()
			
			if mask_num == selected_mask:
				# SELECTED - Use same bright style as hover
				match category:
					"medical":
						style.bg_color = Color(0.25, 0.5, 0.35, 0.6)  # Brighter green
						style.border_color = Color(0.4, 0.8, 0.55, 0.9)  # Bright green border
					"utility":
						style.bg_color = Color(0.45, 0.3, 0.2, 0.6)  # Brighter orange
						style.border_color = Color(1.0, 0.65, 0.4, 0.9)  # Bright orange border
					"cultural":
						style.bg_color = Color(0.35, 0.25, 0.45, 0.6)  # Brighter purple
						style.border_color = Color(0.9, 0.6, 1.0, 0.9)  # Bright purple border
					_:
						style.bg_color = Color(1.0, 1.0, 1.0, 0.3)
						style.border_color = Color(1.0, 1.0, 1.0, 1.0)
				
				style.set_border_width_all(3)  # Thick border like hover
				style.shadow_color = Color(0, 0, 0, 0.5)
				style.shadow_size = 8
				style.shadow_offset = Vector2(0, 4)
			else:
				# UNSELECTED - Use original category style
				match category:
					"medical":
						style.bg_color = Color(0.15, 0.3, 0.2, 0.3)  # Light green tint
						style.border_color = Color(0.3, 0.6, 0.4, 0.6)  # Green border
					"utility":
						style.bg_color = Color(0.3, 0.2, 0.15, 0.3)
						style.border_color = Color(0.8, 0.5, 0.3, 0.6)
					"cultural":
						style.bg_color = Color(0.25, 0.15, 0.3, 0.3)
						style.border_color = Color(0.7, 0.4, 0.8, 0.6)
					_:
						style.bg_color = Color(1.0, 1.0, 1.0, 0.15)
						style.border_color = Color(1.0, 1.0, 1.0, 0.8)
				
				style.set_border_width_all(2)  # Normal border
				style.shadow_color = Color(0, 0, 0, 0.3)
				style.shadow_size = 4
				style.shadow_offset = Vector2(0, 2)
			
			# Apply corner radius and shadow
			style.corner_radius_top_left = 10
			style.corner_radius_top_right = 10
			style.corner_radius_bottom_left = 10
			style.corner_radius_bottom_right = 10
			
			container.add_theme_stylebox_override("panel", style)

func _on_mask_title_button_pressed():
	"""Toggle mask preview expanded/collapsed state."""
	preview_expanded = !preview_expanded
	expandable_content.visible = preview_expanded
	
	# Update button text and panel size
	if preview_expanded:
		# Expand: change arrow to down and resize panel
		mask_preview_panel.custom_minimum_size.y = 260
	else:
		# Collapse: change arrow to right and resize panel  
		mask_preview_panel.custom_minimum_size.y = 50
	
	# Re-update mask preview to refresh arrow icon in title
	if current_mask_num > 0:
		update_mask_preview(current_mask_num)
	else:
		# No mask selected
		var arrow = "▼" if preview_expanded else "▶"
		mask_title_button.text = "%s Pilih Masker" % arrow
	
	print("Mask preview %s" % ("expanded" if preview_expanded else "collapsed"))

func update_mask_preview(mask_num: int):
	"""Update the mask preview title, image, and description with category theming."""
	if mask_num > 0:
		# Get mask info
		var mask_name = mask_names.get(mask_num, "Mask %d" % mask_num)
		var mask_desc = mask_descriptions.get(mask_num, "Deskripsi tidak tersedia")
		
		# Determine category for this mask
		var mask_category = ""
		for category in mask_categories:
			if mask_num in mask_categories[category]:
				mask_category = category
				break
		
		# ALWAYS update title button (regardless of expanded state)
		var arrow = "▼" if preview_expanded else "▶"
		mask_title_button.text = "%s Masker %s" % [arrow, mask_name]
		
		# Update mask preview panel background color based on category
		var panel_bg = mask_preview_panel.get_node("BgStyle") as ColorRect
		match mask_category:
			"medical":
				panel_bg.color = Color(0.12, 0.25, 0.15, 0.75)  # Deep green
			"utility":
				panel_bg.color = Color(0.25, 0.15, 0.1, 0.75)  # Dark orange-brown
			"cultural":
				panel_bg.color = Color(0.2, 0.1, 0.25, 0.75)  # Deep purple
			_:
				panel_bg.color = Color(0, 0, 0, 0.6)  # Default dark
		
		# ALWAYS preload image and description (not just when expanded)
		# This fixes the issue where clicking info after selecting mask shows nothing
		
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
		
		# Update description with category-specific color
		var desc_color = "#E6E6E6"  # Default
		match mask_category:
			"medical":
				desc_color = "#B3FFC8"  # Light green
			"utility":
				desc_color = "#FFD9B3"  # Light orange
			"cultural":
				desc_color = "#E6B3FF"  # Light purple
		
		mask_description.text = "[font_size=14][color=%s]%s[/color][/font_size]" % [desc_color, mask_desc]
	else:
		# No mask active - show placeholder
		var arrow = "▼" if preview_expanded else "▶"
		mask_title_button.text = "%s Pilih Masker" % arrow
		
		# Reset to default background
		var panel_bg = mask_preview_panel.get_node("BgStyle") as ColorRect
		panel_bg.color = Color(0, 0, 0, 0.6)
		
		# Clear content
		mask_preview.texture = null
		mask_description.text = "[font_size=12][color=#B3B3B3]Pilih masker dari kategori di bawah[/color][/font_size]"

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
	"""Handle keyboard input using direct key detection."""
	
	# Quit (Q or ESC)
	if Input.is_key_pressed(KEY_Q) or Input.is_key_pressed(KEY_ESCAPE):
		print("Quit requested from Godot client")
		send_command("quit")
		await get_tree().create_timer(0.1).timeout  # Small delay to send command
		get_tree().quit()
	
	# Toggle Mask (M)
	if Input.is_key_pressed(KEY_M):
		mask_enabled = not mask_enabled
		print("Toggle mask: %s" % ("ON" if mask_enabled else "OFF"))
		send_command("toggle_mask")
		# Add small delay to prevent multiple toggles
		await get_tree().create_timer(0.2).timeout
	
	# Screenshot (S)
	if Input.is_key_pressed(KEY_S):
		print("Screenshot requested")
		send_command("screenshot")
		await get_tree().create_timer(0.2).timeout
	
	# Switch Mask (1-9) using direct key detection
	for i in range(1, 10):
		var key_code = KEY_1 + (i - 1)  # KEY_1, KEY_2, ..., KEY_9
		if Input.is_key_pressed(key_code):
			current_mask_num = i
			print("Switch to mask%d" % i)
			send_command("mask_%d" % i)
			update_mask_preview(i)
			# Add small delay to prevent multiple switches
			await get_tree().create_timer(0.2).timeout
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

func _on_3d_rotation_toggled(enabled: bool):
	"""Toggle 3D rotation on/off."""
	var command = "enable_3d_rotation" if enabled else "disable_3d_rotation"
	send_command(command)
	print("3D Rotation: %s" % ("ENABLED" if enabled else "DISABLED"))
