from typing import List, Dict, Any
from fuzzy import FuzzyTriangleGate


class FTS:
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        luas_tanam: List[float],
        luas_panen: List[float],
        options: Dict[str, Any],
    ):
        self._dataset = dataset
        self._luas_tanam = luas_tanam
        self._luas_panen = luas_panen
        self._margin_multiplier = options.get("marginMultiplier", 0.1)
        self._min_margin = options.get(
            "minMargin", self.min_value * self._margin_multiplier
        )
        self._max_margin = options.get(
            "maxMargin", self.max_value * self._margin_multiplier
        )
        if "interval" in options:
            self._partition_interval = options["interval"]
            self._partition_count = int(
                (self.upper_bound - self.lower_bound) / options["interval"]
            )
        else:
            self._partition_count = options.get("partitionCount", 10)
            self._partition_interval = (
                self.upper_bound - self.lower_bound
            ) / self._partition_count
        self._partition_ref = []
        self._ruleset = {}
        for i in range(self._partition_count):
            prev_point = (
                0
                if i == 0
                else (self.lower_bound + (self._partition_interval * (i - 1)))
            )
            max_point = self.lower_bound + (self._partition_interval * i)
            next_point = self.lower_bound + (self._partition_interval * (i + 1))

            self._partition_ref.append(
                FuzzyTriangleGate(prev_point, max_point, next_point)
            )
            self._ruleset[i] = []

    @property
    def lower_bound(self) -> float:
        return min(v["value"] for v in self._dataset) - self._min_margin

    @property
    def upper_bound(self) -> float:
        return max(v["value"] for v in self._dataset) + self._max_margin

    @property
    def partition_count(self) -> int:
        return self._partition_count

    @property
    def partition_length(self) -> float:
        return self._partition_interval

    @property
    def max_value(self) -> float:
        return max(v["value"] for v in self._dataset)

    @property
    def min_value(self) -> float:
        return min(v["value"] for v in self._dataset)

    def nearest_partition(self, value: float) -> int:
        degrees = [x.degree(value) for x in self._partition_ref]
        highest_degree = max(degrees)
        return degrees.index(highest_degree)

    def train(self) -> None:
        generated_pattern = [self.nearest_partition(v["value"]) for v in self._dataset]

        for i in range(1, len(generated_pattern)):
            precedent = generated_pattern[i - 1]
            consequent = generated_pattern[i]
            luas_tanam = self._luas_tanam[i]
            luas_panen = self._luas_panen[i]

            self._ruleset[precedent].append((consequent, luas_tanam, luas_panen))

    def test(self, options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        base_data = (
            options["dataset"] if options and "dataset" in options else self._dataset
        )
        predicted = []
        previous_luas_tanam = None
        previous_luas_panen = None

        for i, item in enumerate(base_data):
            key, value = item["key"], item["value"]
            luas_tanam = self._luas_tanam[i]
            luas_panen = self._luas_panen[i]

            partition_index = self.nearest_partition(value)
            partition_rules = self._ruleset.get(partition_index, [])

            if not partition_rules:
                predicted_value = self._partition_ref[partition_index].median
            else:
                weighted_sum = 0
                total_weight = 0

                for rule in partition_rules:
                    consequent_value = self._partition_ref[rule[0]].median
                    weight = (rule[1] + rule[2]) / 2
                    weighted_sum += consequent_value * weight
                    total_weight += weight

                predicted_value = weighted_sum / total_weight #predict value

            # Consider the influence of previous luas_tanam and luas_panen
            print(f"{ i }. hasil Prediksi awal: {predicted_value}")
            print("total_weight", total_weight)
            print("total weighted sum", weighted_sum)

            print("value", value)

            print(
                "hasil Prediksi sebelum",
                predicted_value,
            )
            print("luas panen =", luas_panen, "luas tanam =", luas_tanam)
            total_variable = (luas_tanam + luas_panen) / 2
            predicted_value += total_variable
            print(
                "total variable",
                total_variable,
            )
            print(
                "hasil Prediksi sesudah",
                predicted_value,
            )

            predicted.append({"key": key, "value": value, "predicted": predicted_value})

            previous_luas_tanam = luas_tanam
            previous_luas_panen = luas_panen

        return predicted
