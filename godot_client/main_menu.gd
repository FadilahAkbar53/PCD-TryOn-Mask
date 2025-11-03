extends Control

# Main Menu Script for Virtual Try-On Mask Application

@onready var title_label: Label = $CenterContainer/VBoxContainer/TitleContainer/TitleLabel
@onready var subtitle_label: Label = $CenterContainer/VBoxContainer/TitleContainer/SubtitleLabel
@onready var tryon_button: Button = $CenterContainer/VBoxContainer/MenuContainer/TryOnButton
@onready var about_button: Button = $CenterContainer/VBoxContainer/MenuContainer/AboutButton
@onready var team_button: Button = $CenterContainer/VBoxContainer/MenuContainer/TeamButton
@onready var exit_button: Button = $CenterContainer/VBoxContainer/MenuContainer/ExitButton
@onready var animated_bg: ColorRect = $AnimatedBackground
@onready var version_label: Label = $VersionLabel

# Animation variables
var time_passed := 0.0
var title_original_scale := Vector2.ONE

func _ready():
	print("Main Menu Loaded")
	
	# Connect button signals
	tryon_button.pressed.connect(_on_tryon_button_pressed)
	about_button.pressed.connect(_on_about_button_pressed)
	team_button.pressed.connect(_on_team_button_pressed)
	exit_button.pressed.connect(_on_exit_button_pressed)
	
	# Setup button hover effects
	setup_button_hover_effects()
	
	# Animate buttons in on load
	animate_buttons_in()
	
	# Animate title
	animate_title()

func _process(delta):
	# Subtle background animation
	time_passed += delta
	if animated_bg:
		var shader_material = animated_bg.material as ShaderMaterial
		if shader_material:
			shader_material.set_shader_parameter("time", time_passed)

func animate_buttons_in():
	"""Animate buttons sliding in from the right."""
	var buttons = [tryon_button, about_button, team_button, exit_button]
	var delay = 0.0
	
	for button in buttons:
		# Store original position before animating
		var original_x = button.position.x
		
		# Start off-screen to the right
		button.position.x += 500
		button.modulate.a = 0
		
		# Animate in
		var tween = create_tween()
		tween.set_ease(Tween.EASE_OUT)
		tween.set_trans(Tween.TRANS_BACK)
		
		tween.tween_property(button, "position:x", original_x, 0.6).set_delay(delay)
		tween.parallel().tween_property(button, "modulate:a", 1.0, 0.4).set_delay(delay)
		
		delay += 0.1

func animate_title():
	"""Pulse animation for title."""
	var tween = create_tween()
	tween.set_loops()
	tween.set_ease(Tween.EASE_IN_OUT)
	tween.set_trans(Tween.TRANS_SINE)
	
	tween.tween_property(title_label, "scale", Vector2(1.05, 1.05), 2.0)
	tween.tween_property(title_label, "scale", Vector2(1.0, 1.0), 2.0)

func setup_button_hover_effects():
	"""Add hover effects to all menu buttons."""
	var buttons = [tryon_button, about_button, team_button, exit_button]
	
	for button in buttons:
		# Set pivot to center for smooth scaling
		button.pivot_offset = button.size / 2
		button.mouse_entered.connect(_on_button_hover.bind(button))
		button.mouse_exited.connect(_on_button_unhover.bind(button))

func _on_button_hover(button: Button):
	"""Scale up button and add glow on hover - stays in place."""
	# Kill any existing tweens on this button to prevent overlap
	var existing_tweens = get_tree().get_processed_tweens()
	for tween in existing_tweens:
		if tween.is_valid():
			tween.kill()
	
	var tween = create_tween()
	tween.set_ease(Tween.EASE_OUT)
	tween.set_trans(Tween.TRANS_BACK)
	tween.set_parallel(true)
	
	# Scale from center (using pivot_offset)
	tween.tween_property(button, "scale", Vector2(1.08, 1.08), 0.3)
	
	# Add color modulation for glow effect
	if button == tryon_button:
		tween.tween_property(button, "modulate", Color(1.2, 1.2, 1.4), 0.3)
	elif button == exit_button:
		tween.tween_property(button, "modulate", Color(1.4, 1.1, 1.1), 0.3)
	else:
		tween.tween_property(button, "modulate", Color(1.2, 1.2, 1.2), 0.3)

func _on_button_unhover(button: Button):
	"""Reset button scale and color - stays in place."""
	# Kill any existing tweens on this button to prevent overlap
	var existing_tweens = get_tree().get_processed_tweens()
	for tween in existing_tweens:
		if tween.is_valid():
			tween.kill()
	
	var tween = create_tween()
	tween.set_ease(Tween.EASE_OUT)
	tween.set_trans(Tween.TRANS_BACK)
	tween.set_parallel(true)
	
	tween.tween_property(button, "scale", Vector2(1.0, 1.0), 0.3)
	tween.tween_property(button, "modulate", Color(1.0, 1.0, 1.0), 0.3)

func _on_tryon_button_pressed():
	"""Load Try-On Mask scene with fade transition."""
	print("Loading Try-On Mask scene...")
	fade_transition("res://tryon_client.tscn")

func _on_about_button_pressed():
	"""Load About scene with fade transition."""
	print("Loading About scene...")
	fade_transition("res://about.tscn")

func _on_team_button_pressed():
	"""Load Team scene with fade transition."""
	print("Loading Team scene...")
	fade_transition("res://team.tscn")

func _on_exit_button_pressed():
	"""Exit application with fade out."""
	print("Exiting application...")
	
	var tween = create_tween()
	tween.tween_property(self, "modulate:a", 0.0, 0.5)
	tween.tween_callback(get_tree().quit)

func fade_transition(scene_path: String):
	"""Smooth fade transition to another scene."""
	var tween = create_tween()
	tween.tween_property(self, "modulate:a", 0.0, 0.3)
	tween.tween_callback(func(): get_tree().change_scene_to_file(scene_path))

func _input(event):
	"""Handle keyboard input."""
	if event.is_action_pressed("ui_cancel"):  # ESC key
		_on_exit_button_pressed()
