# Challenges Log

| Date | Who | Problem | Fix | Time lost |
|------|-----|---------|-----|-----------|
| 2026-10-10 | Babar | `splits/class_names.json` listed 37 classes and was missing `Tomato___Tomato_mosaic_virus`, so the KB validator could not cover all tomato classes. | Added the missing class; the file now has 38 classes. | |
| 2026-10-11 | Babar | PlantDoc folder names do not match PlantVillage class names (e.g. "Soyabean leaf", "Bell_pepper leaf spot"), and 10 PlantVillage classes have no PlantDoc counterpart. | Built `field_meta/class_mapping.csv` mapping each PlantDoc folder to the exact PlantVillage name; the 10 unmatched classes are excluded from the lab-vs-field comparison. | |
| 2026-10-11 | Babar | PlantDoc has only 2 usable images for tomato spider mites, short of the 5-per-class target. | Kept both (1 test, 1 adapt) and added extra images to tomato early blight, potato late blight and tomato late blight to reach 150. | |
| 2026-10-11 | Babar | With 5 images per class, an exact 2/3 test split would leave classes with no adapt images. | Split each class at ~1/3 adapt with at least one image in each split, giving 92 test / 58 adapt; documented in the report. | |
| 2026-10-12 | Babar | A link check of `knowledge_base/SOURCES.md` found 5 broken links cited by KB entries and 3 invalid unused sources; some entry text was not backed by any working source. | Replaced the broken links with verified extension fact sheets, removed the invalid sources, and rewrote the affected entries to match their sources. | |
| 2026-10-14 | Anas | Perceptual hashing with naive pairwise comparisons would scale quadratically across 54k images. | Partitioned pHash comparisons by class and applied Union-Find connected component grouping to keep runtime tractable and cluster duplicate leaves into discrete group IDs. | |

