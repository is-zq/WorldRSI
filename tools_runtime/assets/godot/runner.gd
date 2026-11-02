extends Node3D
## Shared capture contract. Codex edits scene.gd for each world and interaction.
var options: Dictionary = {}
var scene: Node3D
var palette: Dictionary
var depth_quad: MeshInstance3D
var depth_material: ShaderMaterial

func _ready() -> void:
	for argument in OS.get_cmdline_user_args():
		var pair := argument.trim_prefix("--").split("=", true, 1)
		if pair.size() == 2:
			options[pair[0]] = pair[1]
	_run.call_deferred()

func read_data(path: String) -> Dictionary:
	if path.is_empty():
		return {}
	return JSON.parse_string(FileAccess.get_file_as_string(path))

func write_data(path: String, data: Variant) -> void:
	var file := FileAccess.open(path, FileAccess.WRITE)
	file.store_string(JSON.stringify(data))

func v3(value: Array) -> Vector3:
	return Vector3(value[0], value[1], value[2])

func array3(value: Vector3) -> Array:
	return [value.x, value.y, value.z]

func descendants(root: Node) -> Array[Node]:
	var result: Array[Node] = [root]
	for child in root.get_children():
		result.append_array(descendants(child))
	return result

func object_root(node: Node) -> Node3D:
	var fallback: Node3D = node if node is MeshInstance3D else null
	while node != scene and node != null:
		if node is Node3D and node.has_meta("instance_id"):
			return node
		if node is CollisionObject3D:
			fallback = node
		node = node.get_parent()
	return fallback

func object_record(root: Node3D) -> Dictionary:
	var transform := root.global_transform
	return {"instance_id": str(root.get_meta("instance_id", scene.get_path_to(root))),
		"path": str(scene.get_path_to(root)), "label": str(root.get_meta("label", root.name)),
		"kind": str(root.get_meta("kind", "dynamic" if root is CharacterBody3D else "static")),
		"visible": root.is_visible_in_tree(),
		"transform": [array3(transform.basis.x), array3(transform.basis.y), array3(transform.basis.z), array3(transform.origin)]}

func capture_objects() -> Dictionary:
	var objects: Dictionary = {}
	for node in descendants(scene):
		if node is MeshInstance3D:
			var root := object_root(node)
			if root != null:
				var object := object_record(root)
				objects[object["instance_id"]] = object
	return objects

func capture(index: int) -> Dictionary:
	var nodes: Dictionary = {}
	for node in descendants(scene):
		if node is Node3D:
			var transform: Transform3D = node.transform
			var item := {"basis": [array3(transform.basis.x), array3(transform.basis.y), array3(transform.basis.z)],
				"origin": array3(transform.origin), "visible": node.visible}
			for key in ["instance_id", "kind", "label"]:
				if node.has_meta(key):
					item[key] = node.get_meta(key)
			if node is Camera3D:
				item["camera_transform"] = [array3(node.global_basis.x), array3(node.global_basis.y), array3(node.global_basis.z), array3(node.global_position)]
				item["fov"] = node.fov
				item["current"] = node.current
				for key in ["near", "far", "projection", "size", "keep_aspect", "frustum_offset", "h_offset", "v_offset"]:
					var value: Variant = node.get(key)
					item[key] = [value.x, value.y] if value is Vector2 else value
			if node is CharacterBody3D or node is RigidBody3D:
				item["velocity"] = array3(node.velocity if node is CharacterBody3D else node.linear_velocity)
			if node is RigidBody3D:
				item["angular_velocity"] = array3(node.angular_velocity)
				item["sleeping"] = node.sleeping
			if node is CollisionShape3D:
				item["disabled"] = node.disabled
			if node is CollisionObject3D:
				item["collision_layer"] = node.collision_layer
				item["collision_mask"] = node.collision_mask
			nodes[str(scene.get_path_to(node))] = item
	var camera := get_viewport().get_camera_3d()
	return {"frame": index, "time": index / 24.0, "nodes": nodes,
		"camera_fov": camera.fov, "custom": scene.export_state(), "objects": capture_objects()}

func restore_camera(node: Camera3D, item: Dictionary) -> void:
	var basis: Array = item["basis"]
	node.transform = Transform3D(Basis(v3(basis[0]), v3(basis[1]), v3(basis[2])), v3(item["origin"]))
	if item.has("camera_transform"):
		var camera_transform: Array = item["camera_transform"]
		var target := Transform3D(Basis(v3(camera_transform[0]), v3(camera_transform[1]), v3(camera_transform[2])), v3(camera_transform[3]))
		if not node.global_transform.is_equal_approx(target):
			node.global_transform = target
	node.fov = item["fov"]
	node.current = item["current"]
	for key in ["near", "far", "projection", "size", "keep_aspect", "frustum_offset", "h_offset", "v_offset"]:
		if item.has(key):
			node.set(key, Vector2(item[key][0], item[key][1]) if key == "frustum_offset" else item[key])

func restore_cameras(snapshot: Dictionary) -> void:
	for path in snapshot["nodes"]:
		var node := scene.get_node_or_null(NodePath(path))
		if node is Camera3D:
			restore_camera(node, snapshot["nodes"][path])

func restore(snapshot: Dictionary) -> void:
	# Rebuild topology first so saved transforms apply to recreated nodes.
	scene.restore_state(snapshot.get("custom", {}))
	for path in snapshot.get("nodes", {}):
		var node := scene.get_node_or_null(NodePath(path))
		if node == null:
			continue
		var item: Dictionary = snapshot["nodes"][path]
		for key in ["instance_id", "kind", "label"]:
			if item.has(key):
				node.set_meta(key, item[key])
		var basis: Array = item["basis"]
		node.transform = Transform3D(Basis(v3(basis[0]), v3(basis[1]), v3(basis[2])), v3(item["origin"]))
		node.visible = item["visible"]
		if node is Camera3D:
			restore_camera(node, item)
		if node is CharacterBody3D and item.has("velocity"):
			node.velocity = v3(item["velocity"])
		if node is RigidBody3D and item.has("velocity"):
			node.linear_velocity = v3(item["velocity"])
		if node is RigidBody3D:
			if item.has("angular_velocity"):
				node.angular_velocity = v3(item["angular_velocity"])
			if item.has("sleeping"):
				node.sleeping = item["sleeping"]
		if node is CollisionShape3D and item.has("disabled"):
			node.disabled = item["disabled"]
		if node is CollisionObject3D:
			for key in ["collision_layer", "collision_mask"]:
				if item.has(key):
					node.set(key, item[key])
	# State restoration may create bodies after the replay pass has started.
	if options.get("mode", "bake") != "bake":
		for node in descendants(scene):
			if node is RigidBody3D:
				node.freeze = true

func validate_scene() -> bool:
	var identities: Dictionary = {}
	for node in descendants(scene):
		if node is MeshInstance3D:
			var root := object_root(node)
			if root != null:
				var object := object_record(root)
				var identity: String = object["instance_id"]
				if identity.is_empty() or (identities.has(identity) and identities[identity] != root):
					push_error("Duplicate or empty object instance ID: " + identity)
					return false
				if object["kind"] not in ["static", "dynamic"]:
					push_error("kind must be static or dynamic: " + identity)
					return false
				identities[identity] = root
			if not node.has_meta("semantic_id") or not palette.has(str(node.get_meta("semantic_id"))):
				push_error("Missing semantic label: " + str(node.get_path()))
				return false
			# Replay-only meshes and field proxies need no Godot colliders.
			# Native dynamic bodies still need an active shape when Godot integrates them.
			if options.get("mode", "bake") == "bake" and root != null and (root is RigidBody3D or root is CharacterBody3D):
				var has_shape := false
				for child in root.get_children():
					if child is CollisionShape3D and child.shape != null and not child.disabled:
						has_shape = true
				if not has_shape:
					push_error("Native dynamic body needs an enabled shape: " + str(root.get_path()))
					return false
	return get_viewport().get_camera_3d() != null

func semantic_mode() -> void:
	var viewport := get_viewport()
	viewport.msaa_3d = Viewport.MSAA_DISABLED
	viewport.screen_space_aa = Viewport.SCREEN_SPACE_AA_DISABLED
	viewport.use_taa = false
	for node in descendants(scene):
		if node is MeshInstance3D:
			var color: Array = palette[str(node.get_meta("semantic_id"))]["color"]
			var material := StandardMaterial3D.new()
			material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
			material.albedo_color = Color8(color[0], color[1], color[2])
			node.material_override = material
			node.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		if node is Light3D:
			node.visible = false
		if node is WorldEnvironment:
			var environment := Environment.new()
			environment.background_mode = Environment.BG_COLOR
			var background: Array = palette["0"]["color"]
			environment.background_color = Color8(background[0], background[1], background[2])
			environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
			environment.tonemap_exposure = 1.0
			node.environment = environment
		if node is Camera3D:
			node.attributes = null
			node.environment = null
	# Encode uncovered pixels with the reserved background code.
	var semantic_environment := Environment.new()
	semantic_environment.background_mode = Environment.BG_COLOR
	var background_color: Array = palette["0"]["color"]
	semantic_environment.background_color = Color8(background_color[0], background_color[1], background_color[2])
	semantic_environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	semantic_environment.tonemap_exposure = 1.0
	scene.get_world_3d().environment = semantic_environment

func surface_identity(node: Node) -> String:
	if node.has_meta("surface_id"):
		return str(node.get_meta("surface_id"))
	var root := object_root(node)
	if root != null:
		return object_record(root)["instance_id"] + "/" + str(root.get_path_to(node))
	var parts: PackedStringArray = []
	while node != scene:
		var label := str(node.name)
		# Anonymous child names contain process-local counters; use their stable sibling slot.
		if label.begins_with("@"):
			label = node.get_class() + "[" + str(node.get_index()) + "]"
		parts.insert(0, label)
		node = node.get_parent()
	return "/".join(parts)

func surface_mode(normals: bool) -> Dictionary:
	# Stable node paths identify each rigid mesh segment; explicit IDs survive reparenting.
	semantic_mode()
	var instances: Dictionary = {}
	var identities: Dictionary = {}
	var code := 1
	for node in descendants(scene):
		if node is MeshInstance3D and node.mesh != null:
			var identity := surface_identity(node)
			if identities.has(identity):
				push_error("Duplicate surface instance ID: " + identity)
				get_tree().quit(1)
				return {}
			identities[identity] = true
			var transform: Transform3D = node.global_transform
			instances[str(code)] = {"surface_id": identity,
				"instance_id": object_record(object_root(node))["instance_id"],
				"geometry_version": str(node.get_meta("geometry_version", 0)) + ":" + str(hash(node.mesh.get_faces())),
				"transform": [array3(transform.basis.x), array3(transform.basis.y), array3(transform.basis.z), array3(transform.origin)]}
			var material := ShaderMaterial.new()
			material.shader = load("res://surface.gdshader")
			material.set_shader_parameter("instance_code", float(code))
			material.set_shader_parameter("normals", normals)
			node.material_override = material
			code += 1
	return instances

func depth_mode() -> Dictionary:
	var viewport := get_viewport()
	viewport.msaa_3d = Viewport.MSAA_DISABLED
	viewport.screen_space_aa = Viewport.SCREEN_SPACE_AA_DISABLED
	viewport.use_taa = false
	var camera := viewport.get_camera_3d()
	# Keep packed data out of exposure and color grading.
	camera.environment = Environment.new()
	camera.environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	camera.attributes = null
	if depth_quad == null:
		depth_quad = MeshInstance3D.new()
		var quad := QuadMesh.new()
		quad.size = Vector2(2, 2)
		depth_quad.mesh = quad
		depth_quad.extra_cull_margin = 16384.0
		depth_quad.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		depth_material = ShaderMaterial.new()
		depth_material.shader = load("res://depth.gdshader")
		depth_material.render_priority = 127
		depth_quad.material_override = depth_material
		add_child(depth_quad)
	depth_quad.global_transform = camera.global_transform
	depth_quad.layers = camera.cull_mask
	depth_material.set_shader_parameter("far_distance", camera.far)
	var projection := camera.get_camera_projection()
	var size := viewport.get_visible_rect().size
	return {"far": camera.far, "near": camera.near,
		"intrinsic": [[projection.x.x * size.x / 2.0, 0, (1.0 - projection.z.x) * size.x / 2.0],
			[0, projection.y.y * size.y / 2.0, (1.0 + projection.z.y) * size.y / 2.0], [0, 0, 1]],
		"camera_transform": [array3(camera.global_basis.x), array3(camera.global_basis.y),
			array3(camera.global_basis.z), array3(camera.global_position)]}

func _run() -> void:
	var output: String = options["output"]
	var mode: String = options.get("mode", "bake")
	if mode not in ["bake", "recapture", "rgb", "semantic", "instance", "normal", "depth"]:
		push_error("Unsupported capture mode: " + mode)
		get_tree().quit(1)
		return
	var count := int(options.get("frames", "1"))
	var substeps := int(options.get("substeps", "1"))
	Engine.physics_ticks_per_second = 24 * substeps
	if mode not in ["bake", "recapture"]:
		var adapter := RenderingServer.get_video_adapter_name()
		var vendor := RenderingServer.get_video_adapter_vendor()
		write_data(output.path_join("renderer.json"), {"adapter": adapter, "vendor": vendor,
			"method": RenderingServer.get_current_rendering_method(),
			"driver": RenderingServer.get_current_rendering_driver_name()})
		var software := adapter.to_lower()
		if options.get("require-gpu", "false") == "true" and (adapter.is_empty() or
				"llvmpipe" in software or "softpipe" in software or "software" in software or "swrast" in software):
			push_error("Hardware GPU renderer required; detected: " + adapter)
			get_tree().quit(1)
			return
	palette = read_data("res://palette.json")
	scene = load("res://scene.gd").new()
	scene.name = "Scene"
	add_child(scene)
	scene.build(read_data(options.get("action", "")))
	if mode == "bake":
		# step() drives scripts while bodies remain in the physics space.
		for node in descendants(scene):
			node.set_process(false)
			node.set_physics_process(false)
	else:
		scene.process_mode = Node.PROCESS_MODE_DISABLED
	await get_tree().physics_frame
	await get_tree().physics_frame
	restore(read_data(options.get("state", "")))
	if mode == "bake" and not validate_scene():
		get_tree().quit(1)
		return
	if mode == "recapture":
		var snapshots: Array = read_data(options["trajectory"])["frames"]
		var captured: Array = []
		for index in range(snapshots.size()):
			restore(snapshots[index])
			if not validate_scene():
				get_tree().quit(1)
				return
			captured.append(capture(index))
		write_data(output.path_join("trajectory.json"), {"fps": 24, "width": 832, "height": 480, "frames": captured})
	elif mode == "bake":
		var frames: Array = []
		for index in range(count):
			if index > 0:
				for tick in range(substeps):
					await get_tree().physics_frame
					scene.step(1.0 / (24.0 * substeps))
			frames.append(capture(index))
		write_data(output.path_join("trajectory.json"), {"fps": 24, "width": 832, "height": 480, "frames": frames})
	else:
		var frames: Array = read_data(options["trajectory"])["frames"]
		var directory := output.path_join(mode)
		DirAccess.make_dir_recursive_absolute(directory)
		for index in range(frames.size()):
			restore(frames[index])
			if not validate_scene():
				get_tree().quit(1)
				return
			if mode == "semantic":
				semantic_mode()
			if mode in ["instance", "normal"]:
				write_data(directory.path_join("%06d.json" % index), surface_mode(mode == "normal"))
			if mode == "depth":
				write_data(directory.path_join("%06d.json" % index), depth_mode())
			await get_tree().process_frame
			await RenderingServer.frame_post_draw
			var error := get_viewport().get_texture().get_image().save_png(directory.path_join("%06d.png" % index))
			if error != OK:
				get_tree().quit(1)
				return
	get_tree().quit()
