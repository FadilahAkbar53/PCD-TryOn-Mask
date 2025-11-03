extends Control

# About Scene - Project Description

@onready var back_button: Button = $MarginContainer/VBoxContainer/BackButton

func _ready():
	print("About Scene Loaded")
	back_button.pressed.connect(_on_back_button_pressed)

func _on_back_button_pressed():
	"""Return to main menu."""
	print("Returning to main menu...")
	get_tree().change_scene_to_file("res://main_menu.tscn")

func _input(event):
	"""Handle keyboard input."""
	if event.is_action_pressed("ui_cancel"):  # ESC key
		_on_back_button_pressed()
