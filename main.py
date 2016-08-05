from bow import bow
from w2v import w2v_average, w2v_clustering
#
# print("Bow baseline...")
# bow('datasets/ABSA16_Restaurants_Ru_Train.xml', 'datasets/ABSA16_Restaurants_Ru_Test.xml', "results/bow_baseline.log", False)
# print("Bow with stemming...")
# bow('datasets/ABSA16_Restaurants_Ru_Train.xml', 'datasets/ABSA16_Restaurants_Ru_Test.xml', "results/bow_stemming.log", True)
# print("W2V average...")
# w2v_average('datasets/ABSA16_Restaurants_Ru_Train.xml', 'datasets/ABSA16_Restaurants_Ru_Test.xml',
#             "models/300-40-10-1e3-wiki_ru-restoran-train16-test16-1kk", "results/w2v_average.log")
# print("W2V clustering...")
# w2v_clustering('datasets/ABSA16_Restaurants_Ru_Train.xml', 'datasets/ABSA16_Restaurants_Ru_Test.xml',
#                "models/300-40-10-1e3-wiki_ru-restoran-train16-test16-1kk", "results/w2v_clustering.log")
#

w2v_average('datasets/ABSA15_Restaurants_Train.xml', 'datasets/ABSA15_Restaurants_Test.xml',
            "models/GoogleNews-vectors-negative300.bin.gz", "results/w2v_average.log")
