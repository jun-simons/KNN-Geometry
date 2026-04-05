// kdtree_viz.cpp
// Builds a CGAL kD-tree on a small 2D dummy dataset and dumps it as a
// Graphviz .dot file 
// Usage: ./kdtree_viz [output.dot]

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Kd_tree.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Kernel = CGAL::Simple_cartesian<double>;
using Point_2 = Kernel::Point_2;
using Traits  = CGAL::Search_traits_2<Kernel>;
using Tree    = CGAL::Kd_tree<Traits>;

int main(int argc, char** argv) {
    // dummy dataset, 2D
    // pick varied points for a mix of splits
    std::vector<Point_2> points = {
        {1.0, 2.0}, {2.5, 4.5}, {3.0, 1.5}, {4.0, 6.0}, {5.0, 3.0},
        {6.5, 7.0}, {7.0, 2.5}, {8.0, 5.0}, {9.0, 8.0}, {1.5, 7.5},
        {3.5, 9.0}, {5.5, 1.0}, {6.0, 4.5}, {7.5, 8.5}, {2.0, 5.5},
        {4.5, 3.5}, {8.5, 1.5}, {9.5, 6.0}, {0.5, 4.0}, {5.0, 8.0},
    };

    // bucket_size=1 forces a full binary tree — every leaf holds exactly one
    // point, which makes the graphviz output actually interesting to look at
    Tree tree(points.begin(), points.end(), /*bucket_size=*/1);
    tree.build();

    std::string out_path = (argc >= 2) ? argv[1] : "kdtree.dot";

    std::ofstream ofs(out_path);
    if (!ofs) {
        std::cerr << "Error: could not open output file: " << out_path << "\n";
        return 1;
    }

    tree.write_graphviz(ofs);

    std::cout << "kD-tree written to: " << out_path << "\n";
    std::cout << "Render with:  dot -Tsvg " << out_path << " -o kdtree.svg\n";
    std::cout << "         or:  dot -Tpng " << out_path << " -o kdtree.png\n";

    return 0;
}
