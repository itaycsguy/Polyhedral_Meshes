import bpy
import numpy as np

bl_info = {
    "name": "Mesh Planarity", 
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
	.	Retrieve object vertices.
	.	input: object.
	.	output: vertices.
	'''
	@staticmethod
	def get_obj_vertices(obj):
		return obj.data.vertices
	
	
	'''
	.	Retrieve face vertices.
	.	input: face.
	.	output: face vertices.
	'''
	@staticmethod
	def get_face_vertices(face):
		return face.vertices
	
	
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
		
	

class OBJECT_OT_realization(bpy.types.Operator):
	## Operator
	## Add-On Variables
	bl_idname = "object.mesh_planarity_operator"
	bl_label = "Mesh Planarity Operator"
		
	## App Variables
	SPACE_DIM = 3


	'''
	.	Preprocessing of fetching active object topology.
	.	input: active object.
	.	output: V, F
	'''
	def _preprocessing(self, obj):
		def _get_corrected_vertices(V_hash):
			V = list()
			for i in range(len(V_hash.keys())):
				V.append(V_hash[i])
			return np.asarray(V, dtype=np.float64)

		V, F = {}, list()
		for face in Blender_Adapter.get_faces(obj):
			Fi = list()
			vertices = Blender_Adapter.get_obj_vertices(obj)
			for idx in Blender_Adapter.get_face_vertices(face):
				vertex = vertices[idx]
				vertex_index = Blender_Adapter.get_vertex_index(vertex)
				Fi.append(vertex_index)
				V[vertex_index] = Blender_Adapter.get_vertex_coordinates(vertex, self.SPACE_DIM)
				
			F.append(Fi)
	        
		V = np.array(_get_corrected_vertices(V), order='F', copy=True, dtype=np.float64)
		F = np.array(F, order='F', copy=True, dtype=np.int32)
		return V, F
	

	'''
	.	Operator Add-on execution entry-point.
	'''
	def execute(self, context):
		obj = Blender_Adapter.get_active_obj()
		assert Blender_Adapter.get_mode(obj) == 'OBJECT', 'OBJECT mode is required.'
		Blender_Adapter.select_obj(obj, True)
		V, F = self._preprocessing(obj)
		
		import planarization as plnr
		assert plnr.is_planar_shape(V, F), '{} input is not planar.'.format(obj.name)
		print('{} input is planar.'.format(obj.name))

		return {'FINISHED'}


def register():
	bpy.utils.register_class(OBJECT_OT_realization)
	
def unregister():
	bpy.utils.unregister_class(OBJECT_OT_realization)
		
 
if __name__ == "__main__":
	register()