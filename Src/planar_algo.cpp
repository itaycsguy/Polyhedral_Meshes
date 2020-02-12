/*
.	Names:		Dr. Roi Porrane & Mr. Itay Guy
.	Purpose:	All examples i propose here have taken from libigl.
.	Date:		30/11/2019
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <igl/cotmatrix.h>
#include <igl/per_face_normals.h>
#include <assert.h> 

#define AFFINE		"affine"
#define PARALLEL	"parallel"
#define VERTICAL	"vertical"

class Utils {
public:

	static void from_vectorization(const Eigen::VectorXd& x, Eigen::MatrixXd& Y, int vec_size) {
		Y.resize(vec_size, x.size() / vec_size);
		Y.setZero();

		for (int i = 0; i < x.size();i+=vec_size) {
			Y.block(0, i, vec_size, 1) = x.block(i, 0, vec_size, 1);
		}
	}

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

	static void normal_per_face_matrix(const Eigen::MatrixXd& V_, const Eigen::MatrixXi& F_, Eigen::MatrixXd& M) {
		igl::per_face_normals(V_, F_, M);

		for (int i = 0; i < M.rows(); i++) {
			M.coeffRef(i, M.cols() - 1) = 1.0f;
		}
	}

	static void centering_matrix(Eigen::MatrixXd& J, int num_vertices) {
		Eigen::MatrixXd I, E;
		I.resize(num_vertices, num_vertices);
		I.setIdentity();
		E.resize(num_vertices, num_vertices);
		E.setOnes();

		J = I - (1.0f / num_vertices) * E;
	}

	static void pseudo_inverse(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd singular_diag(svd.singularValues().asDiagonal());

		int dim = singular_diag.rows() < singular_diag.cols() ? singular_diag.rows() : singular_diag.cols();
		for (int i = 0; i < dim; i++) {
			if (singular_diag.coeffRef(i, i) != 0) {
				singular_diag.coeffRef(i, i) = 1.0f / singular_diag.coeffRef(i, i);
			}
		}
		y = svd.matrixU() * singular_diag * svd.matrixV().transpose();
	}

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

	static void eigenvectors(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
		y = svd.matrixV();
	}

	static void degree_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& D) {
		D.resize(V.rows(), V.rows());
		D.setZero();

		for (int i = 0; i < V.rows();i++) {
			for (int j = 0; j < F.rows();j++) {
				Eigen::VectorXi Fi(F.row(j));
				for (int k = 0; k < Fi.size();k++) {
					if (i == Fi(k)) {
						int neighbor_vertex_left, neighbor_vertex_right;
						if (k == 0) {
							neighbor_vertex_left = Fi(Fi.size() - 1);
							neighbor_vertex_right = Fi(k + 1);
						}
						else if (k == (Fi.size() - 1)) {
							neighbor_vertex_left = Fi(k - 1);
							neighbor_vertex_right = Fi(0);
						}
						else {
							neighbor_vertex_left = Fi(k - 1);
							neighbor_vertex_right = Fi(k + 1);
						}

						D.coeffRef(i, neighbor_vertex_left)++;
						D.coeffRef(i, neighbor_vertex_right)++;
					}
				}
			}
		}
	}

	static void adjacency_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& A) {
		A.resize(V.rows(), V.rows());
		A.setZero();

		for (int i = 0; i < F.rows(); i++) {
			Eigen::VectorXi Fi(F.row(i));
			for (int j = 0; j < Fi.size(); j++) {
				int neighbor_vertex_left, neighbor_vertex_right;
				if (j == 0) {
					neighbor_vertex_left = Fi(Fi.size() - 1);
					neighbor_vertex_right = Fi(j + 1);
				}
				else if (j == (Fi.size() - 1)) {
					neighbor_vertex_left = Fi(j - 1);
					neighbor_vertex_right = Fi(0);
				}
				else {
					neighbor_vertex_left = Fi(j - 1);
					neighbor_vertex_right = Fi(j + 1);
				}

				A.coeffRef(Fi(j), neighbor_vertex_left) = 1;
				A.coeffRef(Fi(j), neighbor_vertex_right) = 1;
			}
		}
	}

	static void laplacian_matrix(const Eigen::MatrixXd& D, const Eigen::MatrixXd& A, Eigen::MatrixXd& L, int dim) {
		Eigen::MatrixXd laplacian(D - A);
		L.resize(laplacian.rows() * dim, laplacian.cols() * dim);
		L.setZero();
		
		for (int i = 0; i < laplacian.rows();i++) {
			for (int j = 0; j < dim;j++) {
				L.block(i + j + dim, j * laplacian.cols(), 1, laplacian.cols()) = laplacian.row(i);
			}
		}
	}
};


class PM_Cases {
public:
	int _affine, _parallel, _vertical;

	PM_Cases(const std::vector<std::string>& types, const Eigen::VectorXi& ids) : _affine(-1), _parallel(-1), _vertical(-1) {
		assert(types.size() > 0 && ids.size() > 0);
		assert(types.size() == ids.size());
		for (int i = 0; i < types.size(); i++) {
			std::string type(types.at(i));
			int color = ids(i);
			assert(color >= 0);

			if (!type.compare(AFFINE))			{ this->_affine = color;	}
			else if (!type.compare(PARALLEL))	{ this->_parallel = color;	}
			else if (!type.compare(VERTICAL))	{ this->_vertical = color;	}
			else {
				assert(!type.compare(AFFINE) || !type.compare(PARALLEL) || !type.compare(VERTICAL));
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


class PM_Solver {
	Eigen::MatrixXi _F;
	Eigen::MatrixXd _V, _M, _J, _N;

private:
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

	void parallel_face(Eigen::MatrixXd& Mi, int i) {
		Eigen::RowVectorXd normal(this->_N.row(i));
		Utils::kronecker_product(this->_J.transpose(), normal, Mi);
	}

	void affine_face(Eigen::MatrixXd& Mi, int i) {
		Eigen::MatrixXd Z, A;
		Utils::face_coordinates_matrix(this->_V, this->_F.row(i), Z);
		this->_affine_matrix(Z, this->_J, A);
		Utils::kronecker_product(A, Eigen::Matrix3d::Identity(), Mi);
	}

	void insert_eq(const Eigen::MatrixXd& Mi, int i) {
		int start = this->_M.rows(), dim = this->_V.cols();
		this->_M.conservativeResize(start + Mi.rows(), Eigen::NoChange);
		this->_M.block(start, 0, Mi.rows(), this->_M.cols()).setZero();

		Eigen::VectorXi Fi(this->_F.row(i));
		for (int idx = 0; idx < Fi.size(); idx++) {
			this->_M.block(start, Fi(idx) * dim, Mi.rows(), dim) = Mi.block(0, idx * dim, Mi.rows(), dim);
		}
	}

	void explore_subspace(Eigen::MatrixXd& w) {
		// Find - eigenvectors((P^t) * L * P) s.t. P = I - (M^t) * ((M * (M^t))^(-1)) * M

		Eigen::MatrixXd MMt(this->_M * this->_M.transpose()), MMt_pinv;
		Utils::pseudo_inverse(MMt, MMt_pinv);

		Eigen::MatrixXd I, Ps(this->_M.transpose()* MMt_pinv * this->_M);
		I.resizeLike(Ps);
		I.setIdentity();
		Eigen::MatrixXd P(I - Ps);

		// Laplacian
		Eigen::MatrixXd D, A, L;
		Utils::degree_matrix(this->_V, this->_F, D);
		Utils::adjacency_matrix(this->_V, this->_F, A);
		Utils::laplacian_matrix(D, A, L, this->_V.cols());

		Eigen::MatrixXd objective(P.transpose() * L * P);
		Utils::eigenvectors(objective, w);
	}
};


// By Value Function
std::vector<Eigen::MatrixXd> realization(Eigen::MatrixXd V_, Eigen::MatrixXi F_, Eigen::VectorXi C_, std::vector<std::string> Types_, Eigen::VectorXi IDs_) {

	// Need an input verification and error correspondencly
	PM_Cases cases(Types_, IDs_);

	PM_Solver solver(V_, F_);
	for (int k = 0, i = 0; i < F_.rows();i++) {
		if (C_(i) == cases.get_affine_id()) {
				// Affine case
				// std::cout << "F[" << i << "] -> Affine Case" << std::endl;
				Eigen::MatrixXd Mi;
				solver.affine_face(Mi, i);
				solver.insert_eq(Mi, i);
				break;
		} else if (C_(i) == cases.get_parallel_id()) {
				// Parallel case
				// std::cout << "F[" << i << "] -> Parallel Case" << std::endl;
				Eigen::MatrixXd Mi;
				solver.parallel_face(Mi, i);
				solver.insert_eq(Mi, i);
				break;
		} else if (C_(i) == cases.get_vertical_id()) {
				// Vertical case
				// std::cout << "F[" << i << "] -> Vertical Case" << std::endl;
				break;
		}
	}
	
	Eigen::MatrixXd w, sol;
	solver.explore_subspace(w);
	
	// Packing to return
	std::vector<Eigen::MatrixXd> PM_space;
	sol = w.transpose();
	for (int i = 0; i < sol.rows();i++) {
		PM_space.push_back(sol.row(i));
	}
	return PM_space;
}


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