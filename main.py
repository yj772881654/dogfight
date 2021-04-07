# -*-coding:Utf-8 -*

# ===========================================================

#              - HARFANG® 3D - www.harfang3d.com

#                    - Python tutorial -

#                   Dogfigth - Main module

# ===========================================================
from util import *
from model import *
import harfang as hg
import platform
from SeaRender import *
from WaterReflection import *
from MathsSupp import *
from Camera_follow import *
from data_converter import *
from Machines import *
import json
from math import radians, degrees
from HUD import *
from Clouds_v2 import *
from debug_displays import *
from ScreenModeRequester import *
import random
import time


CALL_END1=False
CALL_END2=False
# =====================================================================================================
#                                   Main class
# =====================================================================================================

class Main:
	resolution = hg.Vector2(1600, 900)
	antialiasing = 2
	screenMode = hg.FullscreenMonitor1

	main_node = hg.Node()

	controller = None

	scene = None
	camera = None
	satellite_camera = None
	camera_matrix = None
	camera_v_move = hg.Vector3(0, 0, 0)  # Camera velocity for sfx
	fps = None
	sea_render = None
	ligth_sun = None
	ligth_sky = None

	sea_render_script = None
	clouds_render_script = None

	water_reflection = None

	p1_aircraft = None
	p2_aircraft = None

	p1_success = False

	carrier = None
	carrier_radar = None
	island = None

	p1_missiles = [None] * 4
	p2_missiles = [None] * 4
	p1_missiles_smoke_color = hg.Color(1, 1, 1, 1)
	p2_missiles_smoke_color = hg.Color(1, 1, 1, 1)

	p1_targets = []

	bullets = None
	ennemy_bullets = None

	title_font = "assets/fonts/destroy.ttf"
	hud_font = "assets/fonts/Furore.otf"
	texture_hud_plot = None
	texture_noise = None

	fading_cptr = 0
	fading_start_saturation = 0
	fadout_flag = False
	fadout_cptr = 0

	audio = None
	p1_sfx = None
	p2_sfx = None

	title_music = 0
	title_music_settings = None

	clouds = None
	render_volumetric_clouds = True

	show_debug_displays = False
	display_gui = False

	satellite_view = False

	HSL_postProcess=None
	MotionBlur_postProcess=None
	RadialBlur_postProcess=None

	flag_MotionBlur=False

	radial_blur_strength = 0.5
	deceleration_blur_strength=1/6
	acceleration_blur_strength=1/3

	gun_sight_2D=None

	current_view=None


# =====================================================================================================
#                                   Functions
# =====================================================================================================

def init_game(plus):
	init_scene(plus)
	Aircraft.main_node = Main.main_node

	Main.audio = hg.CreateMixer()
	Main.audio.Open()

	# Clear color alpha = 0
	Main.scene.GetEnvironment().SetBackgroundColor(hg.Color(0, 0, 0, 0))
	# Aircrafts & Cie:
	Main.p1_aircraft = Aircraft("TangoCharly", 1, "aircraft", plus, Main.scene, hg.Vector3(0, 3000, 0),
								hg.Vector3(0, 0, 0))
	Main.p2_aircraft = Aircraft("Zorglub", 2, "ennemyaircraft", plus, Main.scene, hg.Vector3(4000, 3000, 4000),
								hg.Vector3(0, 0, 0))
	Main.carrier = Carrier("Charles_de_Gaules", 1, plus, Main.scene)

	for i in range(4):
		Main.p1_missiles[i] = Missile("sidewinder_" + str(i), 1, plus, Main.scene, Main.audio,
									  "assets/weaponry/enemymissile_low.geo", "assets/weaponry/enemymissile_smoke")
		Main.p2_missiles[i] = Missile("ennemy_sidewinder_" + str(i), 2, plus, Main.scene, Main.audio,
									  "assets/weaponry/enemymissile_low.geo", "assets/weaponry/enemymissile_smoke")

	# Machine guns :
	Main.bullets = Main.p1_aircraft.gun_machine
	Main.ennemy_bullets = Main.p2_aircraft.gun_machine

	Main.p1_aircraft.set_destroyable_targets([Main.p2_aircraft, Main.carrier])
	Main.p2_aircraft.set_destroyable_targets([Main.p1_aircraft, Main.carrier])

	# Fps
	Main.fps = hg.FPSController(0, 0, 0)

	Main.controller=find_controller("xinput.port0")
	Main.scene.Commit()
	Main.scene.WaitCommit()
	#plus.UpdateScene(Main.scene)

	load_game_parameters()

	Main.texture_hud_plot_aircraft = plus.LoadTexture("assets/sprites/plot.png")
	Main.texture_hud_plot_missile = plus.LoadTexture("assets/sprites/plot_missile.png")
	Main.texture_hud_plot_ship = plus.LoadTexture("assets/sprites/plot_ship.png")
	Main.texture_hud_plot_dir = plus.LoadTexture("assets/sprites/plot_dir.png")
	Main.texture_noise = plus.LoadTexture("assets/sprites/noise.png")

	# ---- Blend settings:

	renderer = plus.GetRenderer()
	renderer.SetBlendFunc(hg.BlendSrcAlpha, hg.BlendOneMinusSrcAlpha)

	# --- Sfx:
	Main.p1_sfx = AircraftSFX(Main.p1_aircraft)
	Main.p2_sfx = AircraftSFX(Main.p2_aircraft)
	# P2 engine sound is different:
	Main.p2_sfx.set_air_pitch(0.75)
	Main.p2_sfx.set_pc_pitch(1.5)
	Main.p2_sfx.set_turbine_pitch_levels(hg.Vector2(1.5, 2.5))

	# ---- Camera:
	Main.scene.SetCurrentCamera(Main.camera)

	# ---- PostProcess:
	Main.MotionBlur_postProcess=hg.MotionBlurPostProcess()
	Main.HSL_postProcess=hg.HSLPostProcess()
	Main.RadialBlur_postProcess=hg.RadialBlurPostProcess()

	Main.camera.AddComponent(Main.RadialBlur_postProcess)
	Main.camera.AddComponent(Main.HSL_postProcess)

	post_processes_load_parameters()


def set_p1_missiles_smoke_color(color: hg.Color):
	Main.p1_missiles_smoke_color = color
	for missile in Main.p1_missiles:
		missile.set_smoke_solor(color)


def set_p2_missiles_smoke_color(color: hg.Color):
	Main.p2_missiles_smoke_color = color
	for missile in Main.p2_missiles:
		missile.set_smoke_solor(color)


def set_p1_gun_color(color: hg.Color):
	Main.bullets.colors = [color]


def set_p2_gun_color(color: hg.Color):
	Main.ennemy_bullets.colors = [color]

#----------------------------------------------------------------------------------------------
#                                   Load / Save game parameters
#----------------------------------------------------------------------------------------------

def load_scene_parameters(file_name="assets/scripts/scene_parameters.json"):
	json_script = hg.GetFilesystem().FileToString(file_name)
	environment = Main.scene.GetEnvironment()
	if json_script != "":
		script_parameters = json.loads(json_script)
		Main.ligth_sun.GetLight().SetDiffuseColor(list_to_color(script_parameters["sunlight_color"]))
		Main.ligth_sky.GetLight().SetDiffuseColor(list_to_color(script_parameters["skylight_color"]))
		environment.SetAmbientColor(list_to_color(script_parameters["ambient_color"]))
		environment.SetAmbientIntensity(script_parameters["ambient_intensity"])
		Main.render_volumetric_clouds = script_parameters["render_clouds"]


def save_scene_parameters(output_filename="assets/scripts/scene_parameters.json"):
	environment = Main.scene.GetEnvironment()
	script_parameters = {"sunlight_color": color_to_list(Main.ligth_sun.GetLight().GetDiffuseColor()),
		"skylight_color": color_to_list(Main.ligth_sky.GetLight().GetDiffuseColor()),
		"ambient_color": color_to_list(environment.GetAmbientColor()),
		"ambient_intensity": environment.GetAmbientIntensity(), "render_clouds": Main.render_volumetric_clouds
						 }
	json_script = json.dumps(script_parameters, indent=4)
	return hg.GetFilesystem().StringToFile(output_filename, json_script)


def load_game_parameters(file_name="assets/scripts/dogfight.json"):
	json_script = hg.GetFilesystem().FileToString(file_name)
	if json_script != "":
		script_parameters = json.loads(json_script)
		set_p1_missiles_smoke_color(list_to_color(script_parameters["p1_missiles_smoke_color"]))
		set_p2_missiles_smoke_color(list_to_color(script_parameters["p2_missiles_smoke_color"]))
		set_p1_gun_color(list_to_color(script_parameters["p1_gun_color"]))
		set_p2_gun_color(list_to_color(script_parameters["p2_gun_color"]))
		Main.radial_blur_strength=script_parameters["radial_blur_strength"]
		Main.deceleration_blur_strength=script_parameters["deceleration_blur_strength"]
		Main.acceleration_blur_strength=script_parameters["acceleration_blur_strength"]

def save_game_parameters(output_filename="assets/scripts/dogfight.json"):
	script_parameters = {"p1_missiles_smoke_color": color_to_list(Main.p1_missiles_smoke_color),
		"p2_missiles_smoke_color": color_to_list(Main.p2_missiles_smoke_color),
		"p1_gun_color": color_to_list(Main.bullets.colors[0]),
		"p2_gun_color": color_to_list(Main.ennemy_bullets.colors[0]),
		"radial_blur_strength": Main.radial_blur_strength,
		"deceleration_blur_strength": Main.deceleration_blur_strength,
		"acceleration_blur_strength": Main.acceleration_blur_strength
						 }
	json_script = json.dumps(script_parameters, indent=4)
	return hg.GetFilesystem().StringToFile(output_filename, json_script)


def post_processes_save_parameters(output_filename="assets/scripts/post_render.json"):
	script_parameters = {"hue": Main.HSL_postProcess.GetH(),
						 "saturation": Main.HSL_postProcess.GetS(),
						 "value":Main.HSL_postProcess.GetL(),
						 "flag_MotionBlur":Main.flag_MotionBlur,
						 "mb_blur_radius":Main.MotionBlur_postProcess.GetBlurRadius(),
						 "mb_exposure":Main.MotionBlur_postProcess.GetExposure(),
						 "mb_sample_count":Main.MotionBlur_postProcess.GetSampleCount()
						 }
	json_script = json.dumps(script_parameters, indent=4)
	return hg.GetFilesystem().StringToFile(output_filename, json_script)


def post_processes_load_parameters(file_name="assets/scripts/post_render.json"):
	json_script = hg.GetFilesystem().FileToString(file_name)
	if json_script != "":
		script_parameters = json.loads(json_script)
		Main.HSL_postProcess.SetH(script_parameters["hue"])
		Main.HSL_postProcess.SetS(script_parameters["saturation"])
		Main.HSL_postProcess.SetL(script_parameters["value"])

		Main.flag_MotionBlur=script_parameters["flag_MotionBlur"]
		Main.MotionBlur_postProcess.SetBlurRadius(script_parameters["mb_blur_radius"])
		Main.MotionBlur_postProcess.SetExposure(script_parameters["mb_exposure"])
		Main.MotionBlur_postProcess.SetSampleCount(int(script_parameters["mb_sample_count"]))

		Main.camera.RemoveComponent(Main.MotionBlur_postProcess)
		if Main.flag_MotionBlur:
			Main.camera.AddComponent(Main.MotionBlur_postProcess)

def load_fps_matrix(fps):
	pos, rot = load_json_matrix("assets/scripts/camera_position.json")
	if pos is not None and rot is not None:
		fps.Reset(pos, rot)

#============================================================================================================
#                                           Scene & Game setups
#============================================================================================================

def init_scene(plus):
	Main.scene = plus.NewScene()
	Main.camera = plus.AddCamera(Main.scene, hg.Matrix4.TranslationMatrix(hg.Vector3(0, 10, -10)))

	Main.camera.SetName("Camera")
	Main.camera.GetCamera().SetZNear(1.)
	Main.camera.GetCamera().SetZFar(40000)

	plus.LoadScene(Main.scene, "assets/aircraft/aircraft.scn")
	plus.LoadScene(Main.scene, "assets/ennemyaircraft/ennemy_aircraft.scn")
	plus.LoadScene(Main.scene, "assets/aircraft_carrier/aircraft_carrier.scn")
	plus.LoadScene(Main.scene, "assets/island/island.scn")
	plus.LoadScene(Main.scene, "assets/feed_backs/feed_backs.scn")

	init_lights(plus)

	while not Main.scene.IsReady():  # Wait until scene is ready
		#plus.UpdateScene(Main.scene, plus.UpdateClock())
		Main.scene.Commit()
		Main.scene.WaitCommit()

	#for i in range(256):
	#   plus.UpdateScene(Main.scene, plus.UpdateClock())
	#	Main.scene.Commit()
	#	Main.scene.WaitCommit()

	Main.satellite_camera = plus.AddCamera(Main.scene, hg.Matrix4.TranslationMatrix(hg.Vector3(0, 1000, 0)))
	setup_satellite_camera(Main.satellite_camera)

	# ---- Clouds:
	json_script = hg.GetFilesystem().FileToString("assets/scripts/clouds_parameters.json")
	if json_script != "":
		clouds_parameters = json.loads(json_script)
		Main.clouds = Clouds(plus, Main.scene, Main.scene.GetNode("Sun"), Main.resolution, clouds_parameters)

	Main.island = Main.scene.GetNode("island")
	Main.island.GetTransform().SetPosition(hg.Vector3(0, 0, 3000))
	Main.island.GetTransform().SetRotation(hg.Vector3(0, 0, 0))

	Main.sea_render_script = hg.RenderScript("assets/lua_scripts/sea_render.lua")
	Main.sea_render_script.SetEnabled(False)
	Main.sea_render = SeaRender(plus, Main.scene, Main.sea_render_script)
	Main.sea_render.load_json_script()

	Main.sea_render.update_render_script(Main.scene, Main.resolution, hg.time_to_sec_f(plus.GetClock()))
	Main.scene.AddComponent(Main.sea_render_script)

	Main.water_reflection = WaterReflection(plus, Main.scene, Main.resolution, Main.resolution.x / 4)

	#Main.clouds_render_script=hg.LogicScript("assets/lua_scripts/clouds_render.lua")
	#Main.scene.AddComponent(Main.clouds_render_script)

	#plus.UpdateScene(Main.scene)
	Main.scene.Commit()
	Main.scene.WaitCommit()
	load_scene_parameters()


def init_lights(plus):
	# Main light:
	Main.ligth_sun = plus.AddLight(Main.scene, hg.Matrix4.RotationMatrix(hg.Vector3(radians(25), radians(-45), 0)),
								   hg.LightModelLinear)
	Main.ligth_sun.SetName("Sun")
	Main.ligth_sun.GetLight().SetDiffuseColor(hg.Color(255. / 255., 255. / 255., 255. / 255., 1.))

	Main.ligth_sun.GetLight().SetShadow(hg.LightShadowMap)  # Active les ombres portées
	Main.ligth_sun.GetLight().SetShadowRange(100)

	Main.ligth_sun.GetLight().SetDiffuseIntensity(1.)
	Main.ligth_sun.GetLight().SetSpecularIntensity(1.)

	# Sky ligth:
	Main.ligth_sky = plus.AddLight(Main.scene, hg.Matrix4.RotationMatrix(hg.Vector3(radians(54), radians(135), 0)),
								   hg.LightModelLinear)
	Main.ligth_sky.SetName("SkyLigth")
	Main.ligth_sky.GetLight().SetDiffuseColor(hg.Color(103. / 255., 157. / 255., 141. / 255., 1.))
	Main.ligth_sky.GetLight().SetDiffuseIntensity(0.5)

	# Ambient:
	environment = hg.Environment()
	environment.SetAmbientColor(hg.Color(103. / 255., 157. / 255., 141. / 255., 1.))
	environment.SetAmbientIntensity(0.5)
	Main.scene.AddComponent(environment)


def find_controller(name):
	nl=name.lower()
	devices = hg.GetInputSystem().GetDevices()
	s = devices.size()
	for i in range(0, s):
		device_id = devices.at(i)
		d=device_id.lower()
		if d == nl:
			return hg.GetInputSystem().GetDevice(device_id)
	return None

#============================================================================================================
#                                          ImGui Interfaces
#============================================================================================================

def gui_interface_scene(scene, fps):
	camera = scene.GetNode("Camera")

	l1_color = Main.ligth_sun.GetLight().GetDiffuseColor()
	l2_color = Main.ligth_sky.GetLight().GetDiffuseColor()
	environment = scene.GetEnvironment()
	amb_color = environment.GetAmbientColor()
	amb_intensity = environment.GetAmbientIntensity()

	if hg.ImGuiBegin("Scene Settings"):
		if hg.ImGuiButton("Load scene parameters"):
			load_scene_parameters()
		hg.ImGuiSameLine()
		if hg.ImGuiButton("Save scene parameters"):
			save_scene_parameters()

		d, f = hg.ImGuiCheckbox("Display collisions shapes", Main.show_debug_displays)
		if d:
			Main.show_debug_displays = f
			scene.GetPhysicSystem().SetDebugVisuals(Main.show_debug_displays)


		d, f = hg.ImGuiCheckbox("Volumetric clouds", Main.render_volumetric_clouds)
		if d:
			Main.render_volumetric_clouds = f
			if not f:
				Main.clouds.clear_particles()

		pos = camera.GetTransform().GetPosition()
		hg.ImGuiText("Camera X " + str(pos.x))
		hg.ImGuiText("Camera Y " + str(pos.y))
		hg.ImGuiText("Camera Z " + str(pos.z))
		if hg.ImGuiButton("Load camera"):
			# load_fps_matrix(fps)
			pos, rot = load_json_matrix("assets/scripts/camera_position.json")
			camera.GetTransform().SetPosition(pos)
			camera.GetTransform().SetRotation(rot)
		hg.ImGuiSameLine()
		if hg.ImGuiButton("Save camera"):
			save_json_matrix(camera.GetTransform().GetPosition(), camera.GetTransform().GetRotation(),
							 "assets/scripts/camera_position.json")

		if hg.ImGuiButton("Load aircraft matrix"):
			pos, rot = load_json_matrix("assets/scripts/aircraft_position.json")
			Main.p1_aircraft.reset(pos, rot)
		hg.ImGuiSameLine()
		if hg.ImGuiButton("Save aircraft matrix"):
			nd = Main.p1_aircraft.get_parent_node()
			save_json_matrix(nd.GetTransform().GetPosition(), nd.GetTransform().GetRotation(),
							 "assets/scripts/aircraft_position.json")

		f, c = hg.ImGuiColorEdit("Ambient color", amb_color)
		if f:
			amb_color = hg.Color(c)
			environment.SetAmbientColor(amb_color)
		d, f = hg.ImGuiSliderFloat("Ambient intensity", amb_intensity, 0, 1)
		if d:
			amb_intensity = f
			environment.SetAmbientIntensity(amb_intensity)

		f, c = hg.ImGuiColorEdit("Sunlight color", l1_color)
		if f:
			l1_color = hg.Color(c)
			Main.ligth_sun.GetLight().SetDiffuseColor(l1_color)

		f, c2 = hg.ImGuiColorEdit("Skylight color", l2_color)
		if f:
			l2_color = hg.Color(c2)
			Main.ligth_sky.GetLight().SetDiffuseColor(l2_color)
	hg.ImGuiEnd()


def gui_interface_game(scene):
	if hg.ImGuiBegin("Game Settings"):
		if hg.ImGuiButton("Load game parameters"):
			load_game_parameters()
		hg.ImGuiSameLine()
		if hg.ImGuiButton("Save game parameters"):
			save_game_parameters()

		f, c = hg.ImGuiColorEdit("P1 Missiles smoke color", Main.p1_missiles_smoke_color,
								 hg.ImGuiColorEditFlags_NoAlpha)
		if f: set_p1_missiles_smoke_color(c)

		f, c = hg.ImGuiColorEdit("P2 Missiles smoke color", Main.p2_missiles_smoke_color)
		if f: set_p2_missiles_smoke_color(c)

		f, c = hg.ImGuiColorEdit("P1 gun color", Main.bullets.colors[0])
		if f: set_p1_gun_color(c)

		f, c = hg.ImGuiColorEdit("P2 gun color", Main.ennemy_bullets.colors[0])
		if f: set_p2_gun_color(c)

		hg.ImGuiSeparator()
		d, f = hg.ImGuiSliderFloat("Radial blur strength", Main.radial_blur_strength, 0, 1)
		if d: Main.radial_blur_strength = f
		d, f = hg.ImGuiSliderFloat("Deceleration blur strength", Main.deceleration_blur_strength, 0, 1)
		if d: Main.deceleration_blur_strength = f
		d, f = hg.ImGuiSliderFloat("Acceleration blur strength", Main.acceleration_blur_strength, 0, 1)
		if d: Main.acceleration_blur_strength = f


	hg.ImGuiEnd()


def gui_interface_sea_render(sea_render: SeaRender, scene, fps):
	if hg.ImGuiBegin("Sea & Sky render Settings"):

		if hg.ImGuiButton("Load sea parameters"):
			sea_render.load_json_script()
		hg.ImGuiSameLine()
		if hg.ImGuiButton("Save sea parameters"):
			sea_render.save_json_script()

		d, f = hg.ImGuiCheckbox("Water reflection", sea_render.render_scene_reflection)
		if d:
			sea_render.render_scene_reflection = f

		d, f = hg.ImGuiSliderFloat("Texture North intensity", sea_render.tex_sky_N_intensity, 0, 1)
		if d: sea_render.tex_sky_N_intensity = f
		d, f = hg.ImGuiSliderFloat("Zenith falloff", sea_render.zenith_falloff, 1, 100)
		if d: sea_render.zenith_falloff = f

		f, c = hg.ImGuiColorEdit("Zenith color", sea_render.zenith_color)
		if f: sea_render.zenith_color = c
		f, c = hg.ImGuiColorEdit("Horizon color", sea_render.horizon_N_color)
		if f: sea_render.horizon_N_color = c

		f, c = hg.ImGuiColorEdit("Water color", sea_render.sea_color)
		if f: sea_render.sea_color = c
		f, c = hg.ImGuiColorEdit("Horizon Water color", sea_render.horizon_S_color)
		if f: sea_render.horizon_S_color = c
		f, c = hg.ImGuiColorEdit("Horizon line color", sea_render.horizon_line_color)
		if f: sea_render.horizon_line_color = c

		hg.ImGuiText("3/4 horizon line size: " + str(sea_render.horizon_line_size))

		f, d = hg.ImGuiCheckbox("Sea texture filtering", bool(sea_render.sea_filtering))
		if f:
			sea_render.sea_filtering = int(d)
		hg.ImGuiText("5/6 max filter samples: " + str(sea_render.max_filter_samples))
		hg.ImGuiText("7/8 filter precision: " + str(sea_render.filter_precision))

		hg.ImGuiText("A/Q sea scale x: " + str(sea_render.sea_scale.x))
		hg.ImGuiText("Z/S sea scale y: " + str(sea_render.sea_scale.y))
		hg.ImGuiText("E/D sea scale z: " + str(sea_render.sea_scale.z))

		d, f = hg.ImGuiSliderFloat("Sea reflection", sea_render.sea_reflection, 0, 1)
		if d: sea_render.sea_reflection = f
		d, f = hg.ImGuiSliderFloat("Reflect offset", Main.sea_render.reflect_offset, 1, 1000)
		if d: Main.sea_render.reflect_offset = f

		hg.ImGuiSeparator()

	hg.ImGuiEnd()


link_altitudes = True
link_morphs = True
clouds_altitude = 1000
clouds_morph_level = 0.1


def gui_clouds(scene: hg.Scene, cloud: Clouds):
	global link_altitudes, link_morphs, clouds_altitude, clouds_morph_level

	if hg.ImGuiBegin("Clouds Settings"):
		if hg.ImGuiButton("Load clouds parameters"):
			cloud.load_json_script()  # fps.Reset(cloud.cam_pos,hg.Vector3(0,0,0))
		hg.ImGuiSameLine()
		if hg.ImGuiButton("Save clouds parameters"):
			cloud.save_json_script(scene)

		hg.ImGuiSeparator()

		hg.ImGuiText("Map position: X=" + str(cloud.map_position.x))
		hg.ImGuiText("Map position: Y=" + str(cloud.map_position.y))

		"""
		d, f = hg.ImGuiSliderFloat("Far Clouds scale x", sky_render.clouds_scale.x, 100, 10000)
		if d:
			sky_render.clouds_scale.x = f

		d, f = hg.ImGuiSliderFloat("Far Clouds scale y", sky_render.clouds_scale.y, 0, 1)
		if d:
			sky_render.clouds_scale.y = f

		d, f = hg.ImGuiSliderFloat("Far Clouds scale z", sky_render.clouds_scale.z, 100, 10000)
		if d:
			sky_render.clouds_scale.z = f


		d, f = hg.ImGuiSliderFloat("Far Clouds absorption", sky_render.clouds_absorption, 0, 1)
		if d:
			sky_render.clouds_absorption = f

		"""

		d, f = hg.ImGuiSliderFloat("Clouds scale x", cloud.map_scale.x, 100, 10000)
		if d:
			cloud.set_map_scale_x(f)
		d, f = hg.ImGuiSliderFloat("Clouds scale z", cloud.map_scale.y, 100, 10000)
		if d:
			cloud.set_map_scale_z(f)

		d, f = hg.ImGuiSliderFloat("Wind speed x", cloud.v_wind.x, -1000, 1000)
		if d:
			cloud.v_wind.x = f

		d, f = hg.ImGuiSliderFloat("Wind speed z", cloud.v_wind.y, -1000, 1000)
		if d:
			cloud.v_wind.y = f

		d, f = hg.ImGuiCheckbox("Link layers altitudes", link_altitudes)
		if d: link_altitudes = f
		d, f = hg.ImGuiCheckbox("Link layers morph levels", link_morphs)
		if d: link_morphs = f

		d, f = hg.ImGuiSliderFloat("Clouds altitude", clouds_altitude, 100, 10000)
		if d:
			clouds_altitude = f
			if link_altitudes:
				for layer in cloud.layers:
					layer.set_altitude(f)

		d, f = hg.ImGuiSliderFloat("Clouds morph level", clouds_morph_level, 0, 1)
		if d:
			clouds_morph_level = f
			if link_morphs:
				for layer in cloud.layers:
					layer.morph_level = f

		for layer in cloud.layers:
			hg.ImGuiSeparator()
			gui_layer(layer)

	hg.ImGuiEnd()


def gui_layer(layer: CloudsLayer):
	nm = layer.name
	hg.ImGuiText(layer.name)

	d, f = hg.ImGuiSliderFloat(nm + " particles rotation speed", layer.particles_rot_speed, -10, 10)
	if d:
		layer.set_particles_rot_speed(f)

	d, f = hg.ImGuiSliderFloat(nm + " particles morph level", layer.morph_level, -1, 1)
	if d:
		layer.morph_level = f

	d, f = hg.ImGuiSliderFloat(nm + " Absorption factor", layer.absorption * 100, 0.01, 10)
	if d:
		layer.set_absorption(f / 100)

	d, f = hg.ImGuiSliderFloat(nm + " Altitude floor", layer.altitude_floor, -2, 2)
	if d: layer.set_altitude_floor(f)

	d, f = hg.ImGuiSliderFloat(nm + " Altitude", layer.altitude, 0, 10000)
	if d: layer.set_altitude(f)

	d, f = hg.ImGuiSliderFloat(nm + " Altitude falloff", layer.altitude_falloff, 0.1, 100)
	if d: layer.set_altitude_falloff(f)

	d, f = hg.ImGuiSliderFloat(nm + " Particles min scale", layer.particles_scale_range.x, 1, 5000)
	if d:
		layer.set_particles_min_scale(f)
	d, f = hg.ImGuiSliderFloat(nm + " Particles max scale", layer.particles_scale_range.y, 1, 5000)
	if d:
		layer.set_particles_max_scale(f)
	d, f = hg.ImGuiSliderFloat(nm + " Alpha threshold", layer.alpha_threshold, 0, 1)
	if d:
		layer.alpha_threshold = f

	d, f = hg.ImGuiSliderFloat(nm + " Scale falloff", layer.scale_falloff, 1, 10)
	if d:
		layer.scale_falloff = f

	d, f = hg.ImGuiSliderFloat(nm + " Alpha scale falloff", layer.alpha_scale_falloff, 1, 10)
	if d:
		layer.alpha_scale_falloff = f

	d, f = hg.ImGuiSliderFloat(nm + " Perturbation", layer.perturbation, 0, 200)
	if d:
		layer.perturbation = f
	d, f = hg.ImGuiSliderFloat(nm + " Tile size", layer.tile_size, 1, 500)
	if d:
		layer.tile_size = f
	d, f = hg.ImGuiSliderFloat(nm + " Distance min", layer.distance_min, 0, 5000)
	if d:
		layer.set_distance_min(f)
	d, f = hg.ImGuiSliderFloat(nm + " Distance max", layer.distance_max, 100, 5000)
	if d:
		layer.set_distance_max(f)

	d, f = hg.ImGuiSliderFloat(nm + " Margin", layer.margin, 0.5, 2)
	if d:
		layer.margin = f
	d, f = hg.ImGuiSliderFloat(nm + " Focal margin", layer.focal_margin, 0.5, 2)
	if d:
		layer.focal_margin = f


def gui_post_rendering():
	if hg.ImGuiBegin("Post-rendering Settings"):
		if hg.ImGuiButton("Load post-render settings"):
			post_processes_load_parameters()
		hg.ImGuiSameLine()
		if hg.ImGuiButton("Save post-render settings"):
			post_processes_save_parameters()

		hg.ImGuiSeparator()

		d, f = hg.ImGuiSliderFloat("Hue", Main.HSL_postProcess.GetH(), -1, 1)
		if d: Main.HSL_postProcess.SetH(f)
		d, f = hg.ImGuiSliderFloat("Saturation", Main.HSL_postProcess.GetS(), 0, 1)
		if d: Main.HSL_postProcess.SetS(f)
		d, f = hg.ImGuiSliderFloat("Luminance", Main.HSL_postProcess.GetL(), 0, 1)
		if d: Main.HSL_postProcess.SetL(f)

		hg.ImGuiSeparator()

		d, f = hg.ImGuiCheckbox("Motion Blur", Main.flag_MotionBlur)
		if d:
			Main.flag_MotionBlur=f
			if f:
				Main.camera.AddComponent(Main.MotionBlur_postProcess)
			else:
				Main.camera.RemoveComponent(Main.MotionBlur_postProcess)

		if Main.flag_MotionBlur:
			pp=Main.MotionBlur_postProcess
			d, i = hg.ImGuiSliderInt("Blur Radius", pp.GetBlurRadius(), 1, 100)
			if d: pp.SetBlurRadius(i)
			d, f = hg.ImGuiSliderFloat("Exposure", pp.GetExposure(), 0, 10)
			if d : pp.SetExposure(f)
			d, f = hg.ImGuiSliderInt("SampleCount", pp.GetSampleCount(), 1, 100)
			if d : pp.SetSampleCount(f)

	hg.ImGuiEnd()


def gui_device_outputs(dev):
	if hg.ImGuiBegin("Paddle inputs"):
		for i in range(hg.Button0, hg.ButtonLast):
			if dev.IsButtonDown(i):
				hg.ImGuiText("Button" + str(i) + " pressed")

		hg.ImGuiText("InputAxisX: " + str(dev.GetValue(hg.InputAxisX)))
		hg.ImGuiText("InputAxisY: " + str(dev.GetValue(hg.InputAxisY)))
		hg.ImGuiText("InputAxisZ: " + str(dev.GetValue(hg.InputAxisZ)))
		hg.ImGuiText("InputAxisS: " + str(dev.GetValue(hg.InputAxisS)))
		hg.ImGuiText("InputAxisT: " + str(dev.GetValue(hg.InputAxisT)))
		hg.ImGuiText("InputAxisR: " + str(dev.GetValue(hg.InputAxisR)))
		hg.ImGuiText("InputRotX: " + str(dev.GetValue(hg.InputRotX)))
		hg.ImGuiText("InputRotY: " + str(dev.GetValue(hg.InputRotY)))
		hg.ImGuiText("InputRotZ: " + str(dev.GetValue(hg.InputRotZ)))
		hg.ImGuiText("InputRotS: " + str(dev.GetValue(hg.InputRotS)))
		hg.ImGuiText("InputRotT: " + str(dev.GetValue(hg.InputRotT)))
		hg.ImGuiText("InputRotR: " + str(dev.GetValue(hg.InputRotR)))
		hg.ImGuiText("InputButton0: " + str(dev.GetValue(hg.InputButton0)))
		hg.ImGuiText("InputButton1: " + str(dev.GetValue(hg.InputButton1)))
		hg.ImGuiText("InputButton2: " + str(dev.GetValue(hg.InputButton2)))
		hg.ImGuiText("InputButton3: " + str(dev.GetValue(hg.InputButton3)))
		hg.ImGuiText("InputButton4: " + str(dev.GetValue(hg.InputButton4)))
		hg.ImGuiText("InputButton5: " + str(dev.GetValue(hg.InputButton5)))
		hg.ImGuiText("InputButton6: " + str(dev.GetValue(hg.InputButton6)))
		hg.ImGuiText("InputButton7: " + str(dev.GetValue(hg.InputButton7)))
		hg.ImGuiText("InputButton8: " + str(dev.GetValue(hg.InputButton8)))
		hg.ImGuiText("InputButton9: " + str(dev.GetValue(hg.InputButton9)))
		hg.ImGuiText("InputButton10: " + str(dev.GetValue(hg.InputButton10)))
		hg.ImGuiText("InputButton11: " + str(dev.GetValue(hg.InputButton11)))
		hg.ImGuiText("InputButton12: " + str(dev.GetValue(hg.InputButton12)))
		hg.ImGuiText("InputButton13: " + str(dev.GetValue(hg.InputButton13)))
		hg.ImGuiText("InputButton14: " + str(dev.GetValue(hg.InputButton14)))
		hg.ImGuiText("InputButton15: " + str(dev.GetValue(hg.InputButton15)))
	hg.ImGuiEnd()

#==================================================================
#           Animations controler
#==================================================================

def animations(dts):
	pass

#===============================================================================================================
#                   Ingame functions
#===============================================================================================================

def autopilot_controller(aircraft: Aircraft):
	if hg.ImGuiBegin("Auto pilot"):
		f, d = hg.ImGuiCheckbox("Autopilot", aircraft.autopilot_activated)
		if f:
			aircraft.autopilot_activated = d

		f, d = hg.ImGuiCheckbox("IA", aircraft.IA_activated)
		if f:
			aircraft.IA_activated = d

		d, f = hg.ImGuiSliderFloat("Pitch", aircraft.autopilot_pitch_attitude, -180, 180)
		if d: aircraft.set_autopilot_pitch_attitude(f)
		d, f = hg.ImGuiSliderFloat("Roll", aircraft.autopilot_roll_attitude, -180, 180)
		if d: aircraft.set_autopilot_roll_attitude(f)
		d, f = hg.ImGuiSliderFloat("Cap", aircraft.autopilot_cap, 0, 360)
		if d: aircraft.set_autopilot_cap(f)
		d, f = hg.ImGuiSliderFloat("Altitude", aircraft.autopilot_altitude, 0, 10000)
		if d: aircraft.set_autopilot_altitude(f)
	hg.ImGuiEnd()


def control_aircraft_paddle(dts, aircraft: Aircraft):

	if Main.controller is None: return False,False,False

	ct = Main.controller

	v = ct.GetValue(hg.InputAxisT)
	if v != 0: aircraft.set_thrust_level(aircraft.thrust_level + v * 0.01)
	if ct.IsButtonDown(hg.ButtonCrossUp): aircraft.set_brake_level(aircraft.brake_level + 0.01)
	if ct.IsButtonDown(hg.ButtonCrossDown): aircraft.set_brake_level(aircraft.brake_level - 0.01)
	if ct.IsButtonDown(hg.ButtonCrossLeft): aircraft.set_flaps_level(aircraft.flaps_level - 0.01)
	if ct.IsButtonDown(hg.ButtonCrossRight): aircraft.set_flaps_level(aircraft.flaps_level + 0.01)

	if ct.WasButtonPressed(hg.Button1):
		if aircraft.post_combution:
			aircraft.deactivate_post_combution()
		else:
			aircraft.activate_post_combution()

	p, y, r = True, True, True
	v = ct.GetValue(hg.InputAxisY)
	if v != 0:
		p = False
		aircraft.set_pitch_level(v)
	v = ct.GetValue(hg.InputAxisX)
	if v != 0:
		r = False
		aircraft.set_roll_level(-v)
	v = -ct.GetValue(hg.InputButton0)
	v += ct.GetValue(hg.InputButton1)
	if v != 0:
		y = False
		aircraft.set_yaw_level(v)

	if ct.IsButtonDown(hg.Button0):
		aircraft.fire_gun_machine()
	elif aircraft.is_gun_activated() and not plus.KeyDown(hg.KeyEnter):
		aircraft.stop_gun_machine()

	if ct.WasButtonPressed(hg.Button2):
		aircraft.fire_missile()

	if ct.WasButtonPressed(hg.Button3):
		aircraft.next_target()

	return p, r, y


def control_aircraft_keyboard(dts, aircraft: Aircraft):
	if plus.KeyDown(hg.KeyLCtrl):
		if plus.KeyPress(hg.KeyHome):
			aircraft.set_thrust_level(aircraft.thrust_level + 0.01)
		if plus.KeyPress(hg.KeyEnd):
			aircraft.set_thrust_level(aircraft.thrust_level - 0.01)
	else:
		if plus.KeyDown(hg.KeyHome):
			aircraft.set_thrust_level(aircraft.thrust_level + 0.01)
		if plus.KeyDown(hg.KeyEnd):
			aircraft.set_thrust_level(aircraft.thrust_level - 0.01)

	if plus.KeyDown(hg.KeyB):
		aircraft.set_brake_level(aircraft.brake_level + 0.01)
	if plus.KeyDown(hg.KeyN):
		aircraft.set_brake_level(aircraft.brake_level - 0.01)
	if plus.KeyDown(hg.KeyC):
		aircraft.set_flaps_level(aircraft.flaps_level + 0.01)
	if plus.KeyDown(hg.KeyV):
		aircraft.set_flaps_level(aircraft.flaps_level - 0.01)

	if plus.KeyPress(hg.KeySpace):
		if aircraft.post_combution:
			aircraft.deactivate_post_combution()
		else:
			aircraft.activate_post_combution()

	p, y, r = True, True, True
	if plus.KeyDown(hg.KeyLeft):
		aircraft.set_roll_level(1)
		r = False
	elif plus.KeyDown(hg.KeyRight):
		aircraft.set_roll_level(-1)
		r = False

	if plus.KeyDown(hg.KeyUp):
		aircraft.set_pitch_level(1)
		p = False
	elif plus.KeyDown(hg.KeyDown):
		aircraft.set_pitch_level(-1)
		p = False

	if plus.KeyDown(hg.KeySuppr):
		aircraft.set_yaw_level(-1)
		y = False
	elif plus.KeyDown(hg.KeyPageDown):
		aircraft.set_yaw_level(1)
		y = False

	# aircraft.stabilize(dts, p, y, r)

	if plus.KeyDown(hg.KeyEnter):
		aircraft.fire_gun_machine()
	elif aircraft.is_gun_activated():
		aircraft.stop_gun_machine()

	if plus.KeyPress(hg.KeyF1):
		aircraft.fire_missile()

	if aircraft == Main.p1_aircraft:
		if plus.KeyPress(hg.KeyT):
			aircraft.next_target()

		if plus.KeyDown(hg.KeyP):
			aircraft.set_health_level(aircraft.health_level + 0.01)
		if plus.KeyDown(hg.KeyM):
			aircraft.set_health_level(aircraft.health_level - 0.01)
	return p, r, y

def set_view(view):
	Main.current_view=view
	set_track_view(view)

def control_views():
	quit_sv = False
	if plus.KeyDown(hg.KeyNumpad9):
		setup_camera_follow(Main.p2_aircraft.get_parent_node(),
							Main.p2_aircraft.get_parent_node().GetTransform().GetPosition(),
							Main.p2_aircraft.get_parent_node().GetTransform().GetWorld().GetRotationMatrix())
	if plus.KeyDown(hg.KeyNumpad7):
		setup_camera_follow(Main.p1_aircraft.get_parent_node(),
							Main.p1_aircraft.get_parent_node().GetTransform().GetPosition(),
							Main.p1_aircraft.get_parent_node().GetTransform().GetWorld().GetRotationMatrix())
	if plus.KeyDown(hg.KeyNumpad2):
		quit_sv = True
		set_view(back_view)
	elif plus.KeyDown(hg.KeyNumpad8):
		quit_sv = True
		set_view(front_view)
	elif plus.KeyDown(hg.KeyNumpad4):
		quit_sv = True
		set_view(left_view)
	elif plus.KeyDown(hg.KeyNumpad6):
		quit_sv = True
		set_view(right_view)
	elif plus.KeyPress(hg.KeyNumpad5):
		if Main.satellite_view:
			Main.satellite_view = False
			Main.scene.SetCurrentCamera(Main.camera)
		else:
			Main.satellite_view = True
			Main.scene.SetCurrentCamera(Main.satellite_camera)

	if quit_sv and Main.satellite_view:
		Main.satellite_view = False
		Main.scene.SetCurrentCamera(Main.camera)

	if Main.satellite_view:
		if plus.KeyDown(hg.KeyInsert):
			increment_satellite_view_size()
		elif plus.KeyDown(hg.KeyPageUp):
			decrement_satellite_view_size()

#===========================================================================================================
#                           Render flows
#===========================================================================================================



def renderScript_flow(plus, t,dts):
	Main.sea_render_script.SetEnabled(True)
	if Main.sea_render.render_scene_reflection and not Main.satellite_view:
		Main.water_reflection.render(plus, Main.scene, Main.camera)

	Main.sea_render.reflect_map = Main.water_reflection.render_texture
	Main.sea_render.reflect_map_depth = Main.water_reflection.render_depth_texture
	Main.sea_render.update_render_script(Main.scene, Main.resolution, hg.time_to_sec_f(plus.GetClock()))
	# Main.scene.Commit()
	# Main.scene.WaitCommit()
	renderer = plus.GetRenderer()
	renderer.EnableDepthTest(True)
	renderer.EnableDepthWrite(True)
	renderer.EnableBlending(True)
	renderer = plus.GetRenderer()
	renderer.ClearRenderTarget()

	# Volumetric clouds:
	if Main.render_volumetric_clouds:
		Main.clouds.update(t,dts, Main.scene, Main.resolution)
		Main.scene.Commit()
		Main.scene.WaitCommit()

	plus.UpdateScene(Main.scene)

def render_flow(plus,delta_t):
	t = hg.time_to_sec_f(plus.GetClock())
	dts=hg.time_to_sec_f(delta_t)

	renderScript_flow(plus, t,dts)

# ==================================================================================================
#                                           Musics
# ==================================================================================================

def start_music(filename):
	music = Main.audio.LoadSound(filename)
	Main.title_music_settings = hg.MixerChannelState()
	Main.title_music_settings.loop_mode = hg.MixerRepeat
	Main.title_music_settings.volume = 1
	Main.title_music_settings.pitch = 1
	Main.title_music = Main.audio.Start(music, Main.title_music_settings)


def start_title_music():
	start_music("assets/sfx/Odesay.xm")


def start_success_music():
	start_music("assets/sfx/analogik_MINI2.XM")


def start_gameover_music():
	start_music("assets/sfx/analogik_nia.xm")


def stop_music():
	Main.audio.Stop(Main.title_music)


# ==================================================================================================
#       Game phases
# ==================================================================================================


def init_start_phase():
	Main.p1_success = False
	Main.audio.StopAll()
	Main.p1_sfx.stop_engine(Main)
	Main.p2_sfx.stop_engine(Main)

	load_fps_matrix(Main.fps)
	# ...or Camera
	camera = Main.scene.GetNode("Camera")
	pos, rot = load_json_matrix("assets/scripts/camera_position.json")
	camera.GetTransform().SetPosition(pos)
	camera.GetTransform().SetRotation(rot)

	# pos, rot = load_json_matrix("assets/scripts/aircraft_position.json")
	pos, rot = Main.carrier.get_aircraft_start_point()
	pos.y += 2
	# Main.p1_aircraft.reset(pos, rot)
	Main.p1_aircraft.IA_activated = False
	Main.p1_aircraft.reset(hg.Vector3( -1.3688265085220337,40,100),
						   rot)
	# On aircraft carrier:
	# pos.z += 40
	# pos.x -= 5
	# Main.p2_aircraft.reset(pos, rot)

	# On flight:

	# Main.p2_aircraft.reset(hg.Vector3(uniform(10000, -10000), uniform(1000, 7000), uniform(10000, -10000)),
	# 					   hg.Vector3(0, radians(uniform(-180, 180)), 0))
	Main.p2_aircraft.reset(hg.Vector3(9500, 6000, 5000),
						   hg.Vector3(0, radians(uniform(-180, 180)), 0))
	Main.p2_aircraft.set_thrust_level(1)

	Main.p1_sfx.reset()
	Main.p2_sfx.reset()

	#plus.UpdateScene(Main.scene)
	Main.scene.Commit()
	Main.scene.WaitCommit()

	setup_camera_follow(Main.p1_aircraft.get_parent_node(),
						Main.p1_aircraft.get_parent_node().GetTransform().GetPosition(),
						Main.p1_aircraft.get_parent_node().GetTransform().GetWorld().GetRotationMatrix())

	Main.scene.SetCurrentCamera(Main.camera)
	Main.satellite_view = False

	Main.HSL_postProcess.SetS(0)
	Main.HSL_postProcess.SetL(0)


	#plus.UpdateScene(Main.scene)
	Main.scene.Commit()
	Main.scene.WaitCommit()
	Main.fading_cptr = 0

	plus.SetFont(Main.title_font)
	for i in range(4):
		Main.p1_aircraft.fit_missile(Main.p1_missiles[i], i)
		Main.p2_aircraft.fit_missile(Main.p2_missiles[i], i)

	Main.audio.SetMasterVolume(1)
	start_title_music()

	Main.RadialBlur_postProcess.SetStrength(0)

	return start_phase


def start_phase(plus, delta_t):
	dts = hg.time_to_sec_f(delta_t)
	camera = Main.scene.GetNode("Camera")

	# Main.fps.UpdateAndApplyToNode(camera, delta_t)
	Main.camera_matrix = update_camera_follow(camera, dts)
	Main.camera_v_move = camera_move * dts

	# Kinetics:

	animations(dts)
	Main.carrier.update_kinetics(Main.scene, dts)
	Main.p1_aircraft.update_kinetics(Main.scene, dts)
	# Main.p2_aircraft.update_kinetics(Main.scene, dts)

	# fade in:
	fade_in_delay = 1.
	Main.fading_cptr = min(fade_in_delay, Main.fading_cptr + dts)

	Main.HSL_postProcess.SetL(Main.fading_cptr / fade_in_delay)

	if Main.fading_cptr >= fade_in_delay:
		# Start infos:
		f = Main.HSL_postProcess.GetL()
		plus.Text2D(514 / 1600 * Main.resolution.x, 771 / 900 * Main.resolution.y, "GET READY",
					0.08 * Main.resolution.y, hg.Color(1., 0.9, 0.3, 1) * f)

		plus.Text2D(554 / 1600 * Main.resolution.x, 671 / 900 * Main.resolution.y,
					"Ennemy aircraft detected : Shoot it down !", 0.02 * Main.resolution.y,
					hg.Color(1., 0.9, 0.3, 1) * f, Main.hud_font)

		plus.Text2D(640 / 1600 * Main.resolution.x, 591 / 900 * Main.resolution.y, "Hit space or Start",
					0.025 * Main.resolution.y,
					hg.Color(1, 1, 1, (1 + sin(hg.time_to_sec_f(plus.GetClock() * 10))) * 0.5) * f)

		s = 0.015
		x = 470 / 1600 * Main.resolution.x
		y = 350
		c = hg.Color(1., 0.9, 0.3, 1) * f
		# Function
		plus.Text2D(x, y / 900 * Main.resolution.y, "Thrust level", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 20) / 900 * Main.resolution.y, "Pitch", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 40) / 900 * Main.resolution.y, "Roll", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 60) / 900 * Main.resolution.y, "SYaw", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 80) / 900 * Main.resolution.y, "Gun", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 100) / 900 * Main.resolution.y, "Missiles", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 120) / 900 * Main.resolution.y, "Target selection", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 140) / 900 * Main.resolution.y, "Brake", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 160) / 900 * Main.resolution.y, "Flaps", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 180) / 900 * Main.resolution.y, "Post combustion (only thrust=100%)", s * Main.resolution.y,
					c, Main.hud_font)
		plus.Text2D(x, (y - 200) / 900 * Main.resolution.y, "Reset game", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 220) / 900 * Main.resolution.y, "Set View", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 240) / 900 * Main.resolution.y, "Satellite Zoom", s * Main.resolution.y, c, Main.hud_font)
		# Keyboard
		c = hg.Color.White
		x = 815 / 1600 * Main.resolution.x
		plus.Text2D(x, y / 900 * Main.resolution.y, "Home / End", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 20) / 900 * Main.resolution.y, "Up / Down", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 40) / 900 * Main.resolution.y, "Right / Left", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 60) / 900 * Main.resolution.y, "Suppr / Page down", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 80) / 900 * Main.resolution.y, "ENTER", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 100) / 900 * Main.resolution.y, "F1", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 120) / 900 * Main.resolution.y, "T", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 140) / 900 * Main.resolution.y, "B / N", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 160) / 900 * Main.resolution.y, "C / V", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 180) / 900 * Main.resolution.y, "Space", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 200) / 900 * Main.resolution.y, "Backspace", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 220) / 900 * Main.resolution.y, "2/4/8/6/5", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 240) / 900 * Main.resolution.y, "Insert / Page Up", s * Main.resolution.y, c, Main.hud_font)

		# Paddle
		x = 990 / 1600 * Main.resolution.x
		plus.Text2D(x, y / 900 * Main.resolution.y, "Right pad up / down", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 20) / 900 * Main.resolution.y, "Left pad", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 40) / 900 * Main.resolution.y, "Left pad", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 60) / 900 * Main.resolution.y, "RT / LT", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 80) / 900 * Main.resolution.y, "A", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 100) / 900 * Main.resolution.y, "X", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 120) / 900 * Main.resolution.y, "Y", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 140) / 900 * Main.resolution.y, "Cross Up / Down", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 160) / 900 * Main.resolution.y, "Cross Right / LEft", s * Main.resolution.y, c,
					Main.hud_font)
		plus.Text2D(x, (y - 180) / 900 * Main.resolution.y, "B", s * Main.resolution.y, c, Main.hud_font)
		plus.Text2D(x, (y - 200) / 900 * Main.resolution.y, "Back", s * Main.resolution.y, c, Main.hud_font)

		start = False
		if Main.controller is not None:
			if Main.controller.WasButtonPressed(hg.ButtonStart): start = True
		# if plus.KeyPress(hg.KeySpace) or start:
		# # if CALL_END1 or start:
		# 	return init_main_phase()
		# if CALL_END2==True:
		# 	return init_main_phase()

		print("eposide start")
		return init_main_phase()

	# rendering:
	render_flow(plus,delta_t)

	return start_phase


def init_main_phase():
	stop_music()

	Main.HSL_postProcess.SetL(1)
	Main.HSL_postProcess.SetS(0)

	set_view(back_view)

	Main.p1_aircraft.set_thrust_level(1)
	# Main.p1_aircraft.set_brake_level(1)
	Main.p1_aircraft.activate_post_combution()
	Main.p1_aircraft.set_linear_speed(200 / 3.6)
	# p2 on carrier:
	# Main.p2_aircraft.set_thrust_level(0)
	# Main.p2_aircraft.set_flaps_level(0.)
	# Main.p2_aircraft.set_brake_level(0)
	# Main.p2_aircraft.activate_post_combution()

	# p2 on flight:
	Main.p2_aircraft.set_linear_speed(800 / 3.6)
	Main.p2_aircraft.IA_activated = True

	Main.p1_aircraft.targets = [Main.p2_aircraft]
	Main.p2_aircraft.targets = [Main.p1_aircraft]
	Main.p2_aircraft.set_target_id(1)

	plus.SetFont(Main.hud_font)
	Main.fading_cptr = 0

	Main.p1_targets = [Main.p2_aircraft, Main.carrier]
	for i in range(len(Main.p1_missiles)):
		Main.p1_targets.append(Main.p1_missiles[i])
		Main.p1_targets.append(Main.p2_missiles[i])

	Main.fadout_flag = False
	Main.fadout_cptr = 0
	return main_phase


def update_radial_post_process(acceleration):
	if acceleration<0:
		Main.RadialBlur_postProcess.SetStrength(Main.radial_blur_strength*pow(min(1, abs(acceleration*Main.deceleration_blur_strength)), 2))
	else:
		Main.RadialBlur_postProcess.SetStrength(Main.radial_blur_strength*pow(min(1, abs(acceleration*Main.acceleration_blur_strength)), 4))

	if Main.gun_sight_2D is not None and Main.current_view==back_view:
		Main.RadialBlur_postProcess.SetCenter(Main.gun_sight_2D)
	else:
		Main.RadialBlur_postProcess.SetCenter(hg.Vector2(0.5,0.5))

def main_phase(plus, delta_t):
	dts = hg.time_to_sec_f(delta_t)
	camera = Main.scene.GetNode("Camera")

	acc=Main.p1_aircraft.get_linear_acceleration()
	noise_level = max(0, Main.p1_aircraft.get_linear_speed() * 3.6 / 2500 * 0.1 + pow(
		min(1, abs(acc / 7)), 2) * 1)
	if Main.p1_aircraft.post_combution:
		noise_level += 0.1

	if Main.satellite_view:
		update_satellite_camera(Main.satellite_camera, Main.resolution.x / Main.resolution.y, dts)
	Main.camera_matrix = update_camera_tracking(camera, dts, noise_level)
	Main.camera_v_move = camera_move * dts
	#Main.fps.UpdateAndApplyToNode(camera, delta_t)
	#Main.camera_matrix = None

	update_radial_post_process(acc)

	# Kinetics:
	fade_in_delay = 1.
	if Main.fading_cptr < fade_in_delay:
		Main.fading_cptr = min(fade_in_delay, Main.fading_cptr + dts)
		Main.HSL_postProcess.SetS( Main.fading_cptr / fade_in_delay * 0.75)

	animations(dts)
	Main.carrier.update_kinetics(Main.scene, dts)
	Main.p1_aircraft.update_kinetics(Main.scene, dts)
	Main.p2_aircraft.update_kinetics(Main.scene, dts)

	pk, rk, yk = control_aircraft_keyboard(dts, Main.p1_aircraft)
	if Main.controller is not None:
		pp, rp, yp = control_aircraft_paddle(dts, Main.p1_aircraft)

	Main.p1_aircraft.stabilize(dts, pk & pp, yk & yp, rk & rp)
	# Hud
	display_hud(Main, plus, Main.p1_aircraft, Main.p1_targets)

	control_views()

	# SFX:
	Main.p1_sfx.update_sfx(Main, dts)
	Main.p2_sfx.update_sfx(Main, dts)

	# rendering:
	render_flow(plus,delta_t)

	if Main.fadout_flag:
		Main.fadout_cptr += dts
		fadout_delay = 1
		f = Main.fadout_cptr / fadout_delay
		Main.audio.SetMasterVolume(1 - f)
		Main.HSL_postProcess.SetL(max(0, 1 - f))
		if Main.fadout_cptr > fadout_delay:
			Main.HSL_postProcess.SetL(0)
			return init_start_phase()

	back = False
	if Main.controller is not None:
		if Main.controller.WasButtonPressed(hg.ButtonBack): back = True
	if plus.KeyPress(hg.KeyBackspace) or back:
		Main.fadout_flag = True

	if Main.p1_aircraft.wreck:
		Main.p1_success = False
		return init_end_phase()

	if Main.p2_aircraft.wreck:
		Main.p1_success = True
		return init_end_phase()

	return main_phase


def init_end_phase():
	Main.fadout_flag = False
	plus.SetFont(Main.title_font)
	Main.fading_cptr = 0
	Main.p1_aircraft.IA_activated = True
	Main.scene.SetCurrentCamera(Main.camera)
	Main.satellite_view = False
	Main.fading_start_saturation = Main.HSL_postProcess.GetS()
	if Main.p1_success:
		start_success_music()
	else:
		start_gameover_music()

	Main.RadialBlur_postProcess.SetStrength(0)

	return end_phase


def end_phase(plus, delta_t):
	dts = hg.time_to_sec_f(delta_t)

	camera = Main.scene.GetNode("Camera")
	Main.camera_matrix = update_camera_follow(camera, dts)
	Main.camera_v_move = camera_move * dts

	# Kinetics:
	if Main.p1_success:
		msg = "MISSION SUCCESSFUL !"
		x = 435 / 1600
		fade_in_delay = 10
		s = 50 / 900
	else:
		msg = "R.I.P."
		x = 685 / 1600
		fade_in_delay = 1
		s = 72 / 900
	if Main.fading_cptr < fade_in_delay:
		Main.fading_cptr = min(fade_in_delay, Main.fading_cptr + dts)
		Main.HSL_postProcess.SetS((1 - Main.fading_cptr / fade_in_delay) * Main.fading_start_saturation)

	animations(dts)
	Main.carrier.update_kinetics(Main.scene, dts)
	Main.p1_aircraft.update_kinetics(Main.scene, dts)
	Main.p2_aircraft.update_kinetics(Main.scene, dts)

	# Hud

	f = Main.HSL_postProcess.GetL()
	plus.Text2D(x * Main.resolution.x, 611 / 900 * Main.resolution.y, msg, s * Main.resolution.y,
				hg.Color(1., 0.9, 0.3, 1) * f)

	plus.Text2D(645 / 1600 * Main.resolution.x, 531 / 900 * Main.resolution.y, "Hit Space or Start",
				0.025 * Main.resolution.y,
				hg.Color(1, 1, 1, (1 + sin(hg.time_to_sec_f(plus.GetClock() * 10))) * 0.5) * f)

	# SFX:
	Main.p1_sfx.update_sfx(Main, dts)
	Main.p2_sfx.update_sfx(Main, dts)

	# rendering:
	render_flow(plus,delta_t)

	start = False
	if Main.controller is not None:
		if Main.controller.WasButtonPressed(hg.ButtonStart): start = True

	if plus.KeyPress(hg.KeySpace) or start:
		Main.fadout_flag = True
		Main.fadout_cptr = 0

	if CALL_END1==True:
		Main.fadout_flag = True
		Main.fadout_cptr = 0


	if Main.fadout_flag:
		Main.fadout_cptr += dts
		fadout_delay = 1
		f = Main.fadout_cptr / fadout_delay
		Main.audio.SetMasterVolume(1 - f)
		Main.HSL_postProcess.SetL(max(0, 1 - f))
		if Main.fadout_cptr > fadout_delay:
			Main.HSL_postProcess.SetL(0)
			return init_start_phase()

	return end_phase


# ==================================================================================================

#                                   Program start here

# ==================================================================================================

plus = hg.GetPlus()
hg.LoadPlugins()
hg.MountFileDriver(hg.StdFileDriver())

# hg.SetLogIsDetailed(True)
# hg.SetLogLevel(hg.LogAll)
smr_ok,scr_mod,scr_res = request_screen_mode()
if smr_ok == "ok":
	Main.resolution.x,Main.resolution.y=scr_res.x,scr_res.y
	Main.screenMode=scr_mod

	plus.RenderInit(int(Main.resolution.x), int(Main.resolution.y), Main.antialiasing, Main.screenMode)
	plus.SetBlend2D(hg.BlendAlpha)
	plus.SetBlend3D(hg.BlendAlpha)
	plus.SetCulling2D(hg.CullNever)

	#------------ Add loading screen -------------
	plus.Clear()
	plus.Flip()
	plus.EndFrame()
	#---------------------------------------------

	init_game(plus)

	#plus.UpdateScene(Main.scene)
	Main.scene.Commit()
	Main.scene.WaitCommit()

	# -----------------------------------------------
	#                   Main loop
	# -----------------------------------------------

	game_phase = init_start_phase()
	game=False
	pregame=False
	index=0
	now=index
	num=0
	FRAME_STEP=15
	end_time=time.time()
	pre_action=0
	end=False
	flag=1
	brain = BrainDQNMain(7)
	step=0
	while not plus.KeyDown(hg.KeyEscape) and not plus.IsAppEnded():
		delta_t = plus.UpdateClock()
		dts = hg.time_to_sec_f(delta_t)

		index+=1

		# print(game,"index ",index)
		if Main.p1_aircraft.wreck==False and Main.p2_aircraft.wreck==False and Main.p1_aircraft.thrust_level==1: #判断游戏是开始
			game=True
		else:
			game=False
		if pregame==False and game==True:  #初始化STATE
			step=0
			pre_state=[Main.p1_aircraft.ground_ray_cast_pos.x,Main.p1_aircraft.ground_ray_cast_pos.z,
										  Main.p1_aircraft.ground_ray_cast_pos.y,Main.p1_aircraft.get_linear_speed() * 3.6,
										  Main.p1_aircraft.health_level,
										  angle_trans(Main.p1_aircraft.pitch_attitude),
				 					      angle_trans(Main.p1_aircraft.roll_attitude),
										  angle_trans(cal_yaw(Main.p1_aircraft.v_move.x, Main.p1_aircraft.v_move.z)),
										  Main.p1_aircraft.target_lock_t,
										  Main.p2_aircraft.ground_ray_cast_pos.x, Main.p2_aircraft.ground_ray_cast_pos.z,
										  Main.p2_aircraft.ground_ray_cast_pos.y, Main.p2_aircraft.get_linear_speed() * 3.6,
										  Main.p2_aircraft.health_level,
										  angle_trans(Main.p2_aircraft.pitch_attitude),
										  angle_trans(Main.p2_aircraft.roll_attitude),
										  angle_trans(cal_yaw(Main.p2_aircraft.v_move.x, Main.p2_aircraft.v_move.z)),
										  Main.p2_aircraft.target_lock_t]
		if (Main.p1_aircraft.wreck or Main.p2_aircraft.wreck) and flag==1:  #结束STATE
			end = True
			flag=0
			new_state = [Main.p1_aircraft.ground_ray_cast_pos.x, Main.p1_aircraft.ground_ray_cast_pos.z,
						 Main.p1_aircraft.ground_ray_cast_pos.y, Main.p1_aircraft.get_linear_speed() * 3.6,
						 Main.p1_aircraft.health_level,
						 angle_trans(Main.p1_aircraft.pitch_attitude),
						 angle_trans(Main.p1_aircraft.roll_attitude),
						 angle_trans(cal_yaw(Main.p1_aircraft.v_move.x, Main.p1_aircraft.v_move.z)),
						 Main.p1_aircraft.target_lock_t,
						 Main.p2_aircraft.ground_ray_cast_pos.x, Main.p2_aircraft.ground_ray_cast_pos.z,
						 Main.p2_aircraft.ground_ray_cast_pos.y, Main.p2_aircraft.get_linear_speed() * 3.6,
						 Main.p2_aircraft.health_level,
						 angle_trans(Main.p2_aircraft.pitch_attitude),
						 angle_trans(Main.p2_aircraft.roll_attitude),
						 angle_trans(cal_yaw(Main.p2_aircraft.v_move.x, Main.p2_aircraft.v_move.z)),
						 Main.p2_aircraft.target_lock_t]
			# print(pre_state)
			# print(new_state)
			# print(end)
			# print(pre_action)

			New = AircraftState(new_state)
			Old = AircraftState(pre_state)
			brain.setPerception(new_state, pre_action, action_judge(Old, New), end)
		else:
			end = False


		# if plus.KeyPress(hg.KeyDown):
		# 	print("munual action")
		# 	action=2
		# 	brain.setPerception(new_state, action, action_judge(Old, New), end)

		if game and index%FRAME_STEP==0:		#每个关键帧
			# step+=1
			# print(step)
			new_state=[Main.p1_aircraft.ground_ray_cast_pos.x,Main.p1_aircraft.ground_ray_cast_pos.z,
										  Main.p1_aircraft.ground_ray_cast_pos.y,Main.p1_aircraft.get_linear_speed() * 3.6,
										  Main.p1_aircraft.health_level,
										  angle_trans(Main.p1_aircraft.pitch_attitude),
				 					      angle_trans(Main.p1_aircraft.roll_attitude),
										  angle_trans(cal_yaw(Main.p1_aircraft.v_move.x, Main.p1_aircraft.v_move.z)),
										  Main.p1_aircraft.target_lock_t,
										  Main.p2_aircraft.ground_ray_cast_pos.x, Main.p2_aircraft.ground_ray_cast_pos.z,
										  Main.p2_aircraft.ground_ray_cast_pos.y, Main.p2_aircraft.get_linear_speed() * 3.6,
										  Main.p2_aircraft.health_level,
										  angle_trans(Main.p2_aircraft.pitch_attitude),
										  angle_trans(Main.p2_aircraft.roll_attitude),
										  angle_trans(cal_yaw(Main.p2_aircraft.v_move.x, Main.p2_aircraft.v_move.z)),
										  Main.p2_aircraft.target_lock_t]
			New=AircraftState(new_state)
			Old=AircraftState(pre_state)

			# print(pre_state)
			# print(new_state)
			# print(end)
			# print(pre_action)

			# num = random.randint(0, 8)   #take action
			# action_judge(Old, New)
			# print(action_judge(Old,New))

			brain.setPerception(new_state, pre_action, action_judge(Old, New), end)
			action = brain.getAction()
			action=action.argmax()

			take_action(action,Main.p1_aircraft)

			pre_state=new_state
			pre_action=action
			flag=1  #结束时的flag
		if game and index %FRAME_STEP!=0: #采取动作
			take_action(pre_action,Main.p1_aircraft)

		if ((Main.p1_aircraft.wreck==True or Main.p2_aircraft.wreck==True) and pregame==True) :
			now=index

		else:
			CALL_END1=False
		# print(index-now)
		if index==now+20:
			CALL_END1=True
		# print(game,Main.p1_aircraft.wreck,Main.p2_aircraft.wreck,CALL_END1)





		pregame=game
		if plus.KeyPress(hg.KeyF12):
			if Main.display_gui:
				Main.display_gui = False
			else:
				Main.display_gui = True
		if Main.display_gui:
			hg.ImGuiMouseDrawCursor(True)
			gui_interface_sea_render(Main.sea_render, Main.scene, Main.fps)
			gui_interface_scene(Main.scene, Main.fps)
			gui_interface_game(Main.scene)
			gui_post_rendering()
			gui_clouds(Main.scene, Main.clouds)
		# edition_clavier(Main.sea_render)
		# autopilot_controller(Main.p1_aircraft)
		# Update game state:

		if Main.show_debug_displays:
			DebugDisplays.affiche_vecteur(plus, Main.camera, Main.p1_aircraft.ground_ray_cast_pos,
										  Main.p1_aircraft.ground_ray_cast_dir * Main.p1_aircraft.ground_ray_cast_length,
										  False)

		if plus.KeyDown(hg.KeyK):
			Main.p2_aircraft.start_explosion()
			Main.p2_aircraft.set_health_level(0)

		# Rendering:
		game_phase = game_phase(plus, delta_t)

		# End rendering:
		plus.Flip()
		plus.EndFrame()

	plus.RenderUninit()
