import java.io.*;
import java.util.*;

public class DecisionTree {

    private Node root;
    private List<String[]> data;
    private String[] features;
    private Random random = new Random();

    public DecisionTree(String filename) {
        loadData(filename);
    }

    private void loadData(String filename) {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            data = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                data.add(line.split(","));
            }
            features = new String[]{"sepal length", "sepal width", "petal length", "petal width"};
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private double gini(List<String[]> subset) {
        Map<String, Integer> classCounts = new HashMap<>();
        for (String[] row : subset) {
            String label = row[row.length - 1];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        double impurity = 1.0;
        int totalInstances = subset.size();
        for (int count : classCounts.values()) {
            double prob = count / (double) totalInstances;
            impurity -= prob * prob;
        }
        return impurity;
    }

    private List<List<String[]>> split(List<String[]> dataset, int featureIndex, double threshold) {
        List<String[]> left = new ArrayList<>();
        List<String[]> right = new ArrayList<>();
        for (String[] row : dataset) {
            if (Double.parseDouble(row[featureIndex]) <= threshold) {
                left.add(row);
            } else {
                right.add(row);
            }
        }
        List<List<String[]>> split = new ArrayList<>();
        split.add(left);
        split.add(right);
        return split;
    }

    private double infoGain(List<List<String[]>> split, double currentGini) {
        int totalInstances = split.get(0).size() + split.get(1).size();
        double weightedGini = 0.0;
        for (List<String[]> subset : split) {
            weightedGini += (subset.size() / (double) totalInstances) * gini(subset);
        }
        return currentGini - weightedGini;
    }

    private Node buildTree(List<String[]> dataset, String[] features) {
        double bestGain = 0.0;
        int bestFeatureIndex = -1;
        double bestThreshold = 0.0;
        double currentGini = gini(dataset);
        List<List<String[]>> bestSplit = null;

        for (int i = 0; i < features.length; i++) {
            Set<Double> thresholds = new HashSet<>();
            for (String[] row : dataset) {
                thresholds.add(Double.parseDouble(row[i]));
            }
            for (double threshold : thresholds) {
                List<List<String[]>> split = split(dataset, i, threshold);
                double gain = infoGain(split, currentGini);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeatureIndex = i;
                    bestThreshold = threshold;
                    bestSplit = split;
                }
            }
        }

        if (bestGain == 0) {
            return new Node(majorityClass(dataset));
        }

        System.out.println("Splitting on feature: " + features[bestFeatureIndex] + " with threshold: " + bestThreshold);
        System.out.println("Gini Index of current node: " + currentGini);

        Node node = new Node(features[bestFeatureIndex], bestThreshold);
        node.left = buildTree(bestSplit.get(0), features);
        node.right = buildTree(bestSplit.get(1), features);
        return node;
    }

    private String majorityClass(List<String[]> dataset) {
        Map<String, Integer> classCounts = new HashMap<>();
        for (String[] row : dataset) {
            String label = row[row.length - 1];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        return Collections.max(classCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    public void train() {
        Collections.shuffle(data, random);
        int trainSize = (int) (data.size() * 0.8);
        List<String[]> trainData = data.subList(0, trainSize);
        List<String[]> valData = data.subList(trainSize, data.size());
        root = buildTree(trainData, features);
        evaluate(valData);
    }

    private void evaluate(List<String[]> valData) {
        int correct = 0;
        for (String[] row : valData) {
            if (classify(row).equals(row[row.length - 1])) {
                correct++;
            }
        }
        double accuracy = correct / (double) valData.size();
        System.out.println("Accuracy: " + accuracy);
    }

    public String classify(String[] input) {
        Node node = root;
        while (node.prediction == null) {
            double value = Double.parseDouble(input[Arrays.asList(features).indexOf(node.feature)]);
            if (value <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return node.prediction;
    }

    public static void main(String[] args) {
        DecisionTree tree = new DecisionTree("iris.txt");
        tree.train();

        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter new data for classification (comma-separated): ");
        String input = scanner.nextLine();
        String[] newInstance = input.split(",");
        System.out.println("Classified as: " + tree.classify(newInstance));
        scanner.close();
    }
}