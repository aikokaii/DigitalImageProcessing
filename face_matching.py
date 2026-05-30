import json
import math


# ==================================================
# LOAD DATABASE
# ==================================================
def load_database(path="database_fitur.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================================================
# FEATURE VECTOR
# ==================================================
def feature_vector(data):

    vector = []

    # fitur global
    vector.extend([
        data["eye_distance"],
        data["eye_to_nose_ratio"],
        data["nose_to_mouth_ratio"],
        data["skin_ratio"],
        data["edge_density"],
        data["contour_density"],
        data["eye_region_edge"],
        data["eye_non_skin"],
        data["nose_region_edge"],
        data["mouth_non_skin"],
        data["symmetry_error"]
    ])

    # fitur grid
    vector.extend(data["grid_edge_density"])
    vector.extend(data["grid_skin_ratio"])
    vector.extend(data["grid_intensity"])

    # fitur tekstur
    vector.extend(data["lbp_histogram"])

    return vector


# ==================================================
# EUCLIDEAN DISTANCE
# ==================================================
def euclidean_distance(v1, v2):

    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(v1, v2))
    )


# ==================================================
# MANHATTAN DISTANCE
# ==================================================
def manhattan_distance(v1, v2):

    return sum(
        abs(a - b) for a, b in zip(v1, v2)
    )


# ==================================================
# CORRELATION MATCHING
# ==================================================
def correlation_matching(v1, v2):

    n = len(v1)

    mean1 = sum(v1) / n
    mean2 = sum(v2) / n

    numerator = sum(
        (a - mean1) * (b - mean2)
        for a, b in zip(v1, v2)
    )

    denominator1 = math.sqrt(
        sum((a - mean1) ** 2 for a in v1)
    )

    denominator2 = math.sqrt(
        sum((b - mean2) ** 2 for b in v2)
    )

    if denominator1 == 0 or denominator2 == 0:
        return 0

    return numerator / (denominator1 * denominator2)


# ==================================================
# CONVERT TO SIMILARITY
# ==================================================
def distance_to_similarity(distance):

    return 100 / (1 + distance)


def correlation_to_similarity(corr):

    return ((corr + 1) / 2) * 100


# ==================================================
# WEIGHTED SCORE
# 70% Correlation
# 30% Manhattan
# ==================================================
def weighted_matching(v1, v2,
                      corr_weight=0.7,
                      manhattan_weight=0.3):

    manhattan = manhattan_distance(v1, v2)

    corr = correlation_matching(v1, v2)

    sim_manhattan = distance_to_similarity(
        manhattan
    )

    sim_corr = correlation_to_similarity(
        corr
    )

    score = (
        corr_weight * sim_corr +
        manhattan_weight * sim_manhattan
    )

    return score


# ==================================================
# FEATURE MATCHING
# ==================================================
def feature_matching(
        input_feature,
        database_path="database_fitur.json",
        method="euclidean"):

    database = load_database(database_path)

    input_vector = feature_vector(
        input_feature
    )

    hasil = []

    for nama, samples in database.items():

        scores = []

        for sample in samples:

            db_vector = feature_vector(
                sample
            )

            # -------------------------
            # Euclidean
            # -------------------------
            if method == "euclidean":

                score = euclidean_distance(
                    input_vector,
                    db_vector
                )

            # -------------------------
            # Weighted
            # -------------------------
            elif method == "weighted":

                score = weighted_matching(
                    input_vector,
                    db_vector
                )

            else:

                raise ValueError(
                    "Method harus euclidean atau weighted"
                )

            scores.append(score)

        avg_score = sum(scores) / len(scores)

        hasil.append({
            "nama": nama,
            "score": avg_score
        })

    # sorting
    if method == "euclidean":

        hasil.sort(
            key=lambda x: x["score"]
        )

    else:

        hasil.sort(
            key=lambda x: x["score"],
            reverse=True
        )

    return hasil


# ==================================================
# DISPLAY RESULT
# ==================================================
def print_result(result, method):

    print("\n")

    print("=" * 70)

    if method == "euclidean":
        print("FEATURE MATCHING - EUCLIDEAN DISTANCE")
    else:
        print("FEATURE MATCHING - WEIGHTED SCORE")

    print("=" * 70)

    for i, item in enumerate(result, start=1):

        if method == "euclidean":

            similarity = distance_to_similarity(
                item["score"]
            )

            print(
                f"{i:2d}. "
                f"{item['nama']:20s}"
                f" Distance = {item['score']:.4f}"
                f" | Similarity = {similarity:.2f}%"
            )

        else:

            print(
                f"{i:2d}. "
                f"{item['nama']:20s}"
                f" Weighted Score = {item['score']:.2f}%"
            )


# ==================================================
# MAIN PROGRAM
# ==================================================
if __name__ == "__main__":

    database = load_database()

    # ------------------------------------------------
    # DATA UJI
    # ------------------------------------------------
    # gunakan salah satu sampel dari database
    input_feature = database["aldo"][0]

    # =================================================
    # EUCLIDEAN
    # =================================================
    euclidean_result = feature_matching(
        input_feature,
        method="euclidean"
    )

    print_result(
        euclidean_result,
        "euclidean"
    )

    # =================================================
    # WEIGHTED
    # =================================================
    weighted_result = feature_matching(
        input_feature,
        method="weighted"
    )

    print_result(
        weighted_result,
        "weighted"
    )