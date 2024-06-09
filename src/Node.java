import java.io.*;
import java.util.*;

class Node {
    String feature;
    double threshold;
    Node left;
    Node right;
    String prediction;

    public Node(String feature, double threshold) {
        this.feature = feature;
        this.threshold = threshold;
    }

    public Node(String prediction) {
        this.prediction = prediction;
    }
}
