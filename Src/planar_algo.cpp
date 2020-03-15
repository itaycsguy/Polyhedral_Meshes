/*
.	Names:			Dr. Roi Porrane & Mr. Itay Guy
.	Artical:		"On Linear Spaces of Polyhedral Meshes"
.	Artical-Link:	https://inf.ethz.ch/personal/poranner/papers/linearPM.pdf
.	Date:			03/2020
.	Version:		1.3
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <igl/cotmatrix.h>
#include <igl/per_face_normals.h>
#include <fstream>
#include <assert.h>
#define assertm(exp, msg) assert(((void)msg, exp))

#define AFFINE		"affine"
#define PARALLEL	"parallel"
#define VERTICAL	"vertical"

#define ROOT		std::string("C:\\Users\\itayguy\\Desktop\\")
#define TXT_SUFFIX	std::string(".txt")
#define PRINT_TXT	true


class Utils {
public:
	/*
	.	Generating the Face-Specific-Coordinates as a matrix.
	*/
	static void print_to_file(const Eigen::MatrixXd& matrix, const std::string& name) {
		std::ofstream file(ROOT + name + TXT_SUFFIX);
		if (file.is_open()) {
			file << matrix;
		}
		file.close();
	}

	/*
	.	Generating the Face-Specific-Coordinates as a matrix.
	*/
	static void face_coordinates_matrix(const Eigen::MatrixXd& V_, const Eigen::VectorXi& F_, Eigen::MatrixXd& Z) {
		Z.resize(V_.cols(), F_.size());
		Z.setZero();

		for (int j = 0; j < F_.size(); j++) {
			Eigen::VectorXd v = V_.row(F_.coeffRef(j));
			for (int k = 0; k < V_.cols(); k++) {
				Z.coeffRef(k, j) = v.coeffRef(k);
			}
		}
	}

	/*
	.	Generating Normal-Per-Face and changing it to Homogenous-Coordinates.
	*/
	static void normal_per_face_matrix(const Eigen::MatrixXd& V_, const Eigen::MatrixXi& F_, Eigen::MatrixXd& M) {
		igl::per_face_normals(V_, F_, M);

		for (int i = 0; i < M.rows(); i++) {
			M.coeffRef(i, M.cols() - 1) = 1.0f;
		}
	}

	/*
	.	Generating Centering-Matrix: J = I - (1 / n)*E.
	*/
	static void centering_matrix(Eigen::MatrixXd& J, int num_vertices) {
		Eigen::MatrixXd I, E;
		I.resize(num_vertices, num_vertices);
		I.setIdentity();
		E.resize(num_vertices, num_vertices);
		E.setOnes();

		J = I - (1.0f / num_vertices) * E;
	}

	/*
	.	Operating Pseudo-Inverse using svd technique.
	*/
	static void pseudo_inverse(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd singular_diag(svd.singularValues().asDiagonal());

		int dim = singular_diag.rows() < singular_diag.cols() ? singular_diag.rows() : singular_diag.cols();
		for (int i = 0; i < dim; i++) {
			if (singular_diag.coeffRef(i, i) != 0) {
				singular_diag.coeffRef(i, i) = 1.0f / singular_diag.coeffRef(i, i);
			}
		}
		y = svd.matrixU() * singular_diag * svd.matrixV();
	}

	/*
	.	Operating Kronecker-Product between A and B.
	*/
	static void kronecker_product(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& kronecker_mat) {
		int gap_rows = B.rows(), gap_cols = B.cols();
		int rows = A.rows(), cols = A.cols();

		kronecker_mat.resize(rows * gap_rows, cols * gap_cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				kronecker_mat.block(i * gap_rows, j * gap_cols, gap_rows, gap_cols) = A.coeffRef(i, j) * B;
			}
		}
	}

	/*
	.	Computing Eigen-Vectors of x.
	*/
	static void eigenvectors(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
		y = svd.matrixV();

		#if PRINT_TXT
		Utils::print_to_file(svd.matrixU(), std::string("MatrixU"));
		Utils::print_to_file(svd.singularValues(), std::string("SingularValues"));
		Utils::print_to_file(svd.matrixV(), std::string("MatrixV"));
		#endif
	}

	/*
	.	Computing Degree-Matrix from an Adjacency-Matrix.
	*/
	static void degree_matrix(const Eigen::MatrixXd& A, Eigen::MatrixXd& D) {
		D.resizeLike(A);
		D.setZero();

		for (int i = 0; i < A.rows();i++) {
			D.coeffRef(i, i) = A.row(i).sum();
		}
	}

	/*
	.	Computing Adjacency-Matrix of V, F.
	*/
	static void adjacency_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& A) {
		A.resize(V.rows(), V.rows());
		A.setZero();

		for (int i = 0; i < V.rows(); i++) {
			for (int j = 0; j < F.rows(); j++) {
				Eigen::VectorXi Fi(F.row(j));
				for (int k = 0; k < Fi.size(); k++) {
					if (i == Fi(k)) {
						int vk = (k == (Fi.size() - 1)) ? Fi(0) : Fi(k + 1);
						A.coeffRef(i, vk) = A.coeffRef(i, vk)++;
					}
				}
			}
		}
	}

	/*
	.	Computing Laplacian-Matrix using Degree-Matrix, Adjacency-Matrix.
	*/
	static void laplacian_matrix(const Eigen::MatrixXd& D, const Eigen::MatrixXd& A, Eigen::MatrixXd& L, int dim) {
		Eigen::MatrixXd laplacian(D - A);
		L.resize(laplacian.rows() * dim, laplacian.cols() * dim);
		L.setZero();

		int laplacian_idx = 0;
		for (int i = 0; i < L.rows(); i+=dim) {
			for (int j = 0; j < dim; j++) {
				L.block(i + j, j * laplacian.cols(), 1, laplacian.cols()) = laplacian.row(laplacian_idx);
			}
			laplacian_idx++;
		}
	}
};

/*
.	Class contained all face cases that is expected by the algorithm.
*/
class PM_Cases {
	int _affine, _parallel, _vertical; // Future-TODO: vertical case.

public:
	PM_Cases(const std::vector<std::string>& types, const Eigen::VectorXi& ids) : _affine(-1), _parallel(-1), _vertical(-1) {
		assertm(types.size() > 0 && ids.size() > 0, "Faces types and faces ids are required.");
		assertm(types.size() == ids.size(), "Faces types amount should be equal to faces ids amount.");
		for (int i = 0; i < types.size(); i++) {
			std::string type(types.at(i));
			int color = ids(i);
			assertm(color >= 0, "Illegal face color number.");

			if (!type.compare(AFFINE))			{ this->_affine = color;	}
			else if (!type.compare(PARALLEL))	{ this->_parallel = color;	}
			else if (!type.compare(VERTICAL))	{ this->_vertical = color;	}
			else {
				assertm(!type.compare(AFFINE) || !type.compare(PARALLEL) || !type.compare(VERTICAL), "Affine, Parallel or Vertical faces types are required for the algorithm.");
			}
		}
	}
	~PM_Cases() {};

	int get_affine_id() {
		return this->_affine;
	}

	int get_parallel_id() {
		return this->_parallel;
	}

	int get_vertical_id() {
		return this->_vertical;
	}
};

/*
.	Class contains the solver functionalities by face cases.
*/
class PM_Solver {
private:
	Eigen::MatrixXi _F;
	Eigen::MatrixXd _V, _M, _J, _N;

	// Computing the Face-Affine-Matrix.
	void _affine_matrix(const Eigen::MatrixXd& Z, const Eigen::MatrixXd& J, Eigen::MatrixXd& A) {
		Eigen::MatrixXd ZcPinv, Zc(Z * J);
		Utils::pseudo_inverse(Zc, ZcPinv);
		Eigen::MatrixXd ZcSym(ZcPinv.transpose() * Zc), I;
		I.resizeLike(ZcSym);
		I.setIdentity();
		A = J * (ZcSym - I);
	}

public:
	PM_Solver(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) : _V(V), _F(F) {
		this->_M.resize(0, V.rows() * V.cols());

		Utils::normal_per_face_matrix(V, F, this->_N);
		Utils::centering_matrix(this->_J, F.cols());
	}
	~PM_Solver() {}

	/*
	.	Computing the parallel case by Kronecker-Product between centering matrix and the Face-Normal.
	*/
	void parallel_face(Eigen::MatrixXd& Mi, int i) {
		Utils::kronecker_product(this->_J.transpose(), this->_N.row(i), Mi);
	}

	/*
	.	Computing the affine case by Kronecker-Product between the Identity-Matrix and the Face-Affine-Matrix.
	*/
	void affine_face(Eigen::MatrixXd& Mi, int i) {
		Eigen::MatrixXd Z, A;
		Utils::face_coordinates_matrix(this->_V, this->_F.row(i), Z);
		this->_affine_matrix(Z, this->_J, A);
		Utils::kronecker_product(A, Eigen::Matrix3d::Identity(), Mi);
	}

	/*
	.	Inserting solutions to the final matrix.
	*/
	void insert_eq(const Eigen::MatrixXd& Mi, int i) {
		int start = this->_M.rows(), dim = this->_V.cols();
		this->_M.conservativeResize(start + Mi.rows(), Eigen::NoChange);
		this->_M.block(start, 0, Mi.rows(), this->_M.cols()).setZero();

		Eigen::VectorXi Fi(this->_F.row(i));
		for (int idx = 0; idx < Fi.size(); idx++) {
			this->_M.block(start, Fi(idx) * dim, Mi.rows(), dim) = Mi.block(0, idx * dim, Mi.rows(), dim);
		}
	}

	/*
	.	Exploring the Max-Linear-Subspace given the Mesh topology by solving the optimization problem: 
	.	Find -> eigenvectors((P^t) * L * P / (P^t)*P) s.t. P = I - (M^t) * ((M * (M^t))^(-1)) * M
	*/
	void explore_subspace(Eigen::MatrixXd& w) {

		Eigen::MatrixXd MMt(this->_M * this->_M.transpose()), MMt_pinv;
		Utils::pseudo_inverse(MMt, MMt_pinv);

		Eigen::MatrixXd I, Ps(this->_M.transpose()* MMt_pinv * this->_M);
		I.resizeLike(Ps);
		I.setIdentity();
		Eigen::MatrixXd P(I - Ps);

		// Laplacian optimization for planar faces
		Eigen::MatrixXd A, D, L;
		Utils::adjacency_matrix(this->_V, this->_F, A);
		Utils::degree_matrix(A, D);
		Utils::laplacian_matrix(D, A, L, this->_V.cols());

		Eigen::MatrixXd objective(P.transpose() * L * P), objective_norm_pinv((P.transpose() * P).inverse());
		Utils::eigenvectors(objective * objective_norm_pinv, w);
	}
};


/*
.	An algorithm Entry-Point.
*/
std::vector<Eigen::MatrixXd> realization(Eigen::MatrixXd V_, Eigen::MatrixXi F_, Eigen::VectorXi C_, std::vector<std::string> Types_, Eigen::VectorXi IDs_) {
	// By Value Without Pointers Capabilities.

	PM_Cases cases(Types_, IDs_);
	PM_Solver solver(V_, F_);
	for (int i = 0; i < F_.rows();i++) {
		if (C_(i) == cases.get_affine_id()) {
			// Affine case
			Eigen::MatrixXd Mi;
			solver.affine_face(Mi, i);
			solver.insert_eq(Mi, i);

		} else if (C_(i) == cases.get_parallel_id()) {
			// Parallel case
			Eigen::MatrixXd Mi;
			solver.parallel_face(Mi, i);
			solver.insert_eq(Mi, i);

		} else if (C_(i) == cases.get_vertical_id()) {
			// Vertical case
			assertm(C_(i) == cases.get_vertical_id(), "Vertical faces type is unsupported yet.");
			// Future-TODO: Implement the Vertical case.

		}
	}
	
	Eigen::MatrixXd w, sol;
	solver.explore_subspace(w); // exploring the Max-Linear-Subspace into w vector.
	
	// Packing to return
	std::vector<Eigen::MatrixXd> PM_space;
	sol = w.transpose();
	for (int i = 0; i < sol.rows();i++) {
		PM_space.push_back(sol.row(i));
	}
	return PM_space;
}


// Independent function which checks if some closed shape is planar of not
bool is_planar_shape(Eigen::MatrixXd V_, Eigen::MatrixXi F_) {

	Eigen::MatrixXd N;
	igl::per_face_normals(V_, F_, N);
	bool is_planar = true;
	for (int i = 0; i < F_.rows();i++) {
		Eigen::VectorXi Fi(F_.row(i));
		Eigen::VectorXd n(N.row(i));
		bool is_planar_face = true;
		for (int j = 1; j <= Fi.size(); j++) {
			Eigen::VectorXd x(j == Fi.size() ? V_.row(Fi(0)) - V_.row(Fi(Fi.size() - 1)) : V_.row(Fi(j)) - V_.row(Fi(j - 1)));
			double dot_prod = n(0) * x(0) + n(1) * x(1) + n(2) * x(2);
			is_planar_face = is_planar_face && (dot_prod == 0.0);
		}
		if (!is_planar_face) {
			is_planar = is_planar_face;
			i = F_.rows();
		}
	}
	return is_planar;
}


// CPP Examples:
// =============

#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_corner_normals.h>
#include <external/glad/src/glad.c>
#include <cstdlib>
#include <string>
#include <iostream>

// Constants
#define PYENV "PYTHONHOME"
#define MODEL "cylinder.off"

// Macros
#define GET_PYENV std::getenv(std::string(PYENV).c_str())

// Global variables
Eigen::MatrixXd V_glob;
Eigen::MatrixXi F_glob;

Eigen::MatrixXd N_vertices_glob;
Eigen::MatrixXd N_faces_glob;
Eigen::MatrixXd N_corners_glob;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
	// This function is called every time a keyboard button is pressed

	switch (key) {
	case '1':
		viewer.data().set_normals(N_faces_glob);
		return true;
	case '2':
		viewer.data().set_normals(N_vertices_glob);
		return true;
	case '3':
		viewer.data().set_normals(N_corners_glob);
		return true;
	default: break;
	}
	return false;
}

void plot_mesh() {

	std::string env_p(GET_PYENV);
	if (env_p.empty()) {
		std::cout << "Environment Variable %" << PYENV << "%: not exist." << std::endl;
		return;
	}

	// Load a mesh in OFF format
	igl::readOFF(env_p + "\\include\\data\\" + MODEL, V_glob, F_glob);

	// Plot the mesh
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V_glob, F_glob);
	viewer.launch();
}

void plot_mesh_normals() {

	std::string env_p(GET_PYENV);
	if (env_p.empty()) {
		std::cout << "Environment Variable %" << PYENV << "%: not exist." << std::endl;
		return;
	}

	// Load a mesh in OFF format
	igl::readOFF(env_p + "\\include\\data\\" + MODEL, V_glob, F_glob);

	// Compute per-face normals
	igl::per_face_normals(V_glob, F_glob, N_faces_glob);

	// Compute per-vertex normals
	igl::per_vertex_normals(V_glob, F_glob, N_vertices_glob);

	// Compute per-corner normals, |dihedral angle| > 20 degrees --> crease
	igl::per_corner_normals(V_glob, F_glob, 20, N_corners_glob);

	// Plot the mesh
	igl::opengl::glfw::Viewer viewer;
	viewer.callback_key_down = &key_down;
	viewer.data().show_lines = false;
	viewer.data().set_mesh(V_glob, F_glob);
	viewer.data().set_normals(N_faces_glob);
	std::cout <<
		"Press '1' for per-face normals." << std::endl <<
		"Press '2' for per-vertex normals." << std::endl <<
		"Press '3' for per-corner normals." << std::endl;
	viewer.launch();
}

PYBIND11_MODULE(planarization, m) {

	m.def("realization", &realization, R"pbdoc(
        Realization of a planar mesh.
    )pbdoc");

	m.def("is_planar_shape", &is_planar_shape, R"pbdoc(
        Shape planarity verification.
    )pbdoc");

	m.def("plot_mesh", &plot_mesh, R"pbdoc(
        Plotting a simple mesh.
    )pbdoc");

	m.def("plot_mesh_normals", &plot_mesh_normals, R"pbdoc(
        Plotting a mesh with its normals.
    )pbdoc");

	m.def("print_pylist", [](py::list x) {
		for (auto item : x) {
			py::print(item);
		}
	});

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}