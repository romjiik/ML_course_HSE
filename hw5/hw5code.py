import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    #объединяю фичу и таргет в один вектор
    total_vector = np.column_stack((feature_vector, target_vector))
    total_vector = total_vector[total_vector[:,0].argsort()]
    #выбираю уникальный значения и создаю пороги
    feature_values = np.unique(total_vector[:, 0])
    tresholds = (feature_values[:-1] + feature_values[1:]) / 2
    
    feature_vector_bound = np.append(total_vector[:, 0], total_vector[:, 0][-1])
    feature_vector_bound = np.array(feature_vector_bound)
    
    target = total_vector[:, 1]
    source_node = np.cumsum(np.ones(len(target))) #сколько всего объектов имеется при каждом пороге
    left_node_1 = np.cumsum(target) #сколько объектов положительного класса попало в левый узел
    left_node_0 = source_node - left_node_1  #сколько объектов отрицательного класса попало в левый узел
    left_node = left_node_1 + left_node_0 #всего в левом узле
    #аналогично с правым узлом
    right_node_1 = np.sum(target) - left_node_1
    right_node_0 = (len(target) - np.sum(target)) - left_node_0
    right_node = right_node_1 + right_node_0
    #нахожу коэффициенты
    left_share = left_node / len(target)
    right_share = right_node / len(target)
    H_left = 1 - (left_node_1 / left_node)**2 - (left_node_0 / left_node)**2
    H_right = 1 - (right_node_1 / right_node)**2 - (right_node_0 / right_node)**2
    #считаю джини
    ginis = - left_share * H_left - right_share * H_right
    ginis = ginis[(feature_vector_bound[:-1] - feature_vector_bound[1:]) != 0]
    #нахожу лучшие оценки
    gini_best = np.max(ginis)
    threshold_best = tresholds[np.argmax(ginis)]
    return tresholds, ginis, threshold_best, gini_best
    pass


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]): #проверка на лист тут изменил с != на ==
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]): #поменял на 0
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count #тут наоборот current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))# x[0] в первой лямбда функции
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) #list
            else:
                raise ValueError
            if len(np.unique(feature_vector)) == 1: #поставил 1 для константы
                continue

            _, _, threshold, gini = find_best_split(feature_vector, np.array(sub_y)) 
            if gini_best is None or gini > gini_best:
                feature_best = feature 
                gini_best = gini
                split = feature_vector < threshold 

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": #поменял на маленькую букву
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items()))) 

                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] #добавил [][]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}  
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"]) #добавил logical к y

    def _predict_node(self, x, node):
        
        def go(node):
            if 'threshold' in node:
                if x[node['feature_split']] < node['threshold']:
                    return node['left_child']
                return node['right_child']
            if x[node['feature_split']] in node['categories_split']:
                 return node['left_child']
            return node['right_child']
                
        current_node = go(node)
        while current_node['type'] != 'terminal':
            current_node = go(current_node)
        return current_node['class']
        pass

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
