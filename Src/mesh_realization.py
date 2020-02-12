import bpy
import bmesh
import numpy as np
import random
from bpy.props import FloatProperty

bl_info = {
    "name": "Mesh Realization",
    "blender": (2, 80, 0),
    "category": "Object",
}

class Blender_Adapter:	
	
	'''
	.	Selecting an object.
	.	input: object, boolean selection.
	'''
	@staticmethod
	def select_obj(obj, select):
		obj.select_set(select)
	
	
	'''
	.	Retrieve the active object.
	.	output: active object.
	'''
	@staticmethod
	def get_active_obj():
		return bpy.context.active_object


	'''
	.	Enable showing the object name.
	.	input: object.
	'''
	@staticmethod
	def enable_show_obj_name(obj):
		obj.show_name = True


	'''
	.	Retrieve object name.
	.	input: object.
	.	output: object name.
	'''
	@staticmethod
	def get_obj_name(obj):
		return obj.name


	'''
	.	Retrieve the environment mode.
	.	input: object.
	.	output: mode of work.
	'''
	@staticmethod
	def get_mode(obj):
		return bpy.context.object.mode
		
	
	'''
	.	Retrieve faces.
	.	input: object.
	.	ouput: object faces.
	'''
	@staticmethod
	def get_faces(obj):
		return obj.data.polygons
	
	
	'''
	.	Retrieve face vertices.
	.	input: face.
	.	output: face vertices.
	'''
	@staticmethod
	def get_face_vertices(face):
		return face.vertices
	
	
	'''
	.	Retrieve face material index.
	.	input: face.
	.	output: material index.
	'''
	@staticmethod
	def get_face_material_index(face):
		return face.material_index
	
	
	'''
	.	Retrieve object face material name.
	.	input: object, face.
	.	output: object matrial name in lower case.
	'''
	@staticmethod
	def get_face_material_name(obj, face):
		mi = Blender_Adapter.get_face_material_index(face)
		return obj.data.materials[mi].name.lower()
	
	
	'''
	.	Retrieve vertex coordinate as tuple.
	.	input: vertex, dimensions.
	.	output: coordinates tuple.
	'''
	@staticmethod
	def get_vertex_coordinates(vertex, dim=3):
		vd = list()
		for i in range(dim):
			vd.append(vertex.co[i])	
		return tuple(vd)
	
	
	'''
	.	Retrieve object vertex index.
	.	input: vertex.
	.	output: index.
	'''
	@staticmethod
	def get_vertex_index(vertex):
		return vertex.index
	
	
	'''
	.	Retrieve object vertices.
	.	input: object.
	.	output: vertices.
	'''
	@staticmethod
	def get_obj_vertices(obj):
		return obj.data.vertices


	'''
	.	Setting new vertices.
	.	input: target vertices, source vertices.
	'''
	@staticmethod
	def set_obj_vertices(target, source_tup):
		target.co = source_tup
		

	'''
	.	Setting new vertices.
	.	input: target vertices, source vertices.
	'''
	@staticmethod
	def get_shifted_location(location_vec, shift_vec):
		sloc = list()
		for ax, gap in shift_vec.items():
			sloc.append(location_vec[ax] + gap)
		return tuple(sloc)


	'''
	.	Object generator.
	.	input: object name, data to copy from, reference object to use, gaps in axis.
	.	output: new object.
	'''
	@staticmethod
	def gen_object(name, object_data, ref_obj=None, x_gap=.0, y_gap=.0, z_gap=.0):
		obj = bpy.data.objects.new(name=name, object_data=object_data)
		if ref_obj is not None:
			obj.location = Blender_Adapter.get_shifted_location(ref_obj.location, {0: x_gap, 1: y_gap, 2: z_gap})
		return obj

	
	'''
	.	Links new object to 3D space.
	.	input: object.
	'''
	@staticmethod
	def link_obj(obj):
		bpy.context.collection.objects.link(obj)


	'''
	.	Add blender shape keys to eigen shapes.
	.	input: active object, a list of all eigen shapes.
	'''
	@staticmethod	
	def add_basis_shape_key(obj, ref_shapes, min_glob_sliders=.0, max_glob_sliders=1000.):
		sk = obj.shape_key_add(name=Blender_Adapter.get_obj_name(obj))
		sk.interpolation = 'KEY_LINEAR'
		obj.data.shape_keys.use_relative = True
		obj.hide_viewport = False
		for shape in ref_shapes:
			shape.select_set(True)
		bpy.ops.object.join_shapes()
		for shape in ref_shapes:
			shape.select_set(False)
			shape.hide_set(True)
		for shape_key in bpy.data.shape_keys:
			for key_block in shape_key.key_blocks:
				key_block.slider_min = min_glob_sliders
				key_block.slider_max = max_glob_sliders
	
			
	'''
	.	Add blender shape keys to eigen shapes.
	.	input: active object, a list of all eigen shapes.
	'''
	@staticmethod	
	def add_keyframes(basis_obj_name, ref_ptrn, frames=250, sigma=15.):
		def _get_gauss_smooth_value(i, size, frame, sigma):
			# value = np.exp(-((frame * (size / frames) - i)**2) / ((2 * sigma)**2))
			# value = np.exp(-(frame * (size / frames) - i) / sigma)
			value = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((frame * (size / frames) - i)**2) / ((2 * sigma)**2))
			return value
		
		def _sorted_by_name(key_blocks):
			pair_list = list()
			for i, kb in enumerate(key_blocks):
				if kb.name != basis_obj_name:
					pair_list.append((kb.name, kb))
			return sorted(pair_list, key=lambda x: x[0])
		
		for frame in range(frames):
			for shape_key in bpy.data.shape_keys:
				key_blocks = _sorted_by_name(shape_key.key_blocks)
				for i, kb in enumerate(key_blocks):
					## already sorted running from high freqs. to low freqs.
					kb = kb[1]
					if ref_ptrn in kb.name and kb.name != basis_obj_name:
						kb.value = _get_gauss_smooth_value(i, len(key_blocks), frame, sigma)
						kb.keyframe_insert("value", frame=frame)
			
		

class OBJECT_OT_realization(bpy.types.Operator):
	## Operator
	## Add-On Variables
	bl_idname = "object.mesh_realization_operator"
	bl_label = "Mesh Realization Operator"
	
	## App Variables
	X_GAP, Y_GAP, Z_GAP = 0, 5, 0
	FINAL_Y_GAP = -5
	USE_RELATIVE = True
	INTERPOLATION_METHOD = 'KEY_LINEAR'
	PRODUCED_SHAPE_NAME = 'EIGEN_SHAPE'
	SPACE_DIM = 3
	## Transferring As 2 Vectors To The CPP Code
	FACES_COLOR = {'affine': 0, 'parallel': 1} #, 'vertical': 2}
	

	'''
	.	Preprocessing of fetching active object topology.
	.	input: active object.
	.	output: V, F, C, types, ids
	'''
	def _preprocessing(self, obj):
		def _get_corrected_color_name(target_name, class_colors):
			for color in class_colors:
				if color in target_name:
					return color

		def _get_uniform_face_vertices(F):
			min = len(F[0])
			for i in range(len(F)):
				Sf = len(F[i])
				if Sf < min:
					min = Sf
					
			F_uniform = list()
			for i in range(len(F)):
				Sf = len(F[i])
				if Sf == min:
					F_uniform.append(F[i])
			return F_uniform
		

		V, F, C = {}, list(), list()
		class_colors = self.FACES_COLOR.keys()
		for face in Blender_Adapter.get_faces(obj):
			target_color = Blender_Adapter.get_face_material_name(obj, face)
			material_name = _get_corrected_color_name(target_color, class_colors)
			C.append(self.FACES_COLOR[material_name])
			Fi = list()
			vertices = Blender_Adapter.get_obj_vertices(obj)
			for idx in Blender_Adapter.get_face_vertices(face):
				vertex = vertices[idx]
				vertex_index = Blender_Adapter.get_vertex_index(vertex)
				V[vertex_index] = Blender_Adapter.get_vertex_coordinates(vertex, self.SPACE_DIM)
				Fi.append(vertex_index)
			F.append(Fi)
	        
		V_arr = np.zeros(len(V)).tolist()
		for vi, vv in V.items():
			V_arr[vi] = vv
		V = V_arr
		V = np.array(V, order='F', copy=True, dtype=np.float64)
		F = np.array(_get_uniform_face_vertices(F), order='F', copy=True, dtype=np.int32)
		C = np.array(C, order='F', copy=True, dtype=np.int32)

		TYPES, IDS = list(), list()
		for op, id in self.FACES_COLOR.items():
			TYPES.append(op)
			IDS.append(id)
	    
		TYPES = np.array(TYPES, order='F', copy=True, dtype=np.string_)
		IDS = np.array(IDS, order='F', copy=True, dtype=np.int8)
		return V, F, C, TYPES, IDS


	'''
	.	Parser to the return new vertices into numpy array of arrays to vertices-like form.
	.	input: vertices.
	.	output: vertices.
	'''
	def _parse_vertices(self, V_new):
		V = list()
		for v in V_new:
			V.append(v[0].tolist())
			
		Vco = list()
		for pos in range(0, len(V)):
			sub_Vco = list()
			for i in range(0, len(V[pos]), self.SPACE_DIM):
				vd = list()
				for j in range(self.SPACE_DIM):
					vd.append(V[pos][i + j])	
				sub_Vco.append(tuple(vd))
			Vco.append(sub_Vco)      
		return Vco
	
		
	'''
	.	PM space of eigen shapes generator.
	.	input: active object, new vertices into numpy array of arrays.
	'''
	def _gen_PM_shapes(self, obj, Vco_news):
		Blender_Adapter.enable_show_obj_name(obj)
		basis_obj_name = Blender_Adapter.get_obj_name(obj)
		eigen_shapes = list()
		x_gap, y_gap, z_gap = self.X_GAP, self.Y_GAP, self.Z_GAP
		for i, v_new in enumerate(Vco_news):
			if i == 2:
				## DEBUG
				#pass
				break
			
			shape_name = '{}_{}_{}'.format(basis_obj_name, self.PRODUCED_SHAPE_NAME, i + 1)
			eigen_shape = Blender_Adapter.gen_object(shape_name, obj.data.copy(), ref_obj=obj, x_gap=x_gap, y_gap=y_gap, z_gap=z_gap)
			for curr_v, v in zip(Blender_Adapter.get_obj_vertices(eigen_shape), v_new):
				vd = list()
				for i in range(self.SPACE_DIM):
					vd.append(v[i])
				Blender_Adapter.set_obj_vertices(curr_v, tuple(vd))
			Blender_Adapter.link_obj(eigen_shape)
			eigen_shapes.append(eigen_shape)
			x_gap += self.X_GAP
			y_gap += self.Y_GAP
			z_gap += self.Z_GAP
		Blender_Adapter.add_basis_shape_key(obj, eigen_shapes)
		Blender_Adapter.add_keyframes(Blender_Adapter.get_obj_name(obj), self.PRODUCED_SHAPE_NAME)
		

	'''
	.	Operator Add-on execution entry-point.
	'''
	def execute(self, context):
		obj = Blender_Adapter.get_active_obj()
		assert Blender_Adapter.get_mode(obj) == 'OBJECT', 'OBJECT mode is required.'
		
		Blender_Adapter.select_obj(obj, True)
		V, F, C, TYPES, IDS = self._preprocessing(obj)

		import planarization as plnr
		if plnr.is_planar_shape(V, F):
			print('{} is planar.'.format(obj.name))
		else:
			pass
			#assert False, '{} is not planar.'.format(obj.name)
			
		Vco_news = self._parse_vertices(plnr.realization(V, F, C, TYPES, IDS))
		self._gen_PM_shapes(obj, Vco_news)

		return {'FINISHED'}


def register():
	bpy.utils.register_class(OBJECT_OT_realization)
	
def unregister():
	bpy.utils.unregister_class(OBJECT_OT_realization)
		
 
if __name__ == "__main__":
    register()