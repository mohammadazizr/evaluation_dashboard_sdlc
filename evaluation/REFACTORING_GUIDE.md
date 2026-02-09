# Refactoring Guide - Hierarchical Metrics

## Overview

Kode hierarchical metrics telah di-refactor untuk meningkatkan maintainability dan reduce duplication. Shared logic sudah dipindahkan ke utility folder.

## Struktur Baru

```
evaluation/
├── evaluator.py (✏️ REFACTORED)
├── dashboard_generator.py (✏️ REFACTORED)
├── utils/ (📁 NEW)
│   ├── __init__.py (NEW)
│   └── hierarchical_metrics.py (📁 NEW - Core utilities)
└── [other files...]
```

---

## Utility Classes di `evaluation/utils/hierarchical_metrics.py`

### 1. **HierarchyExtractor**
Menangani extraction dan management path hierarchy

```python
# Extract hierarchy dari path
hierarchy = HierarchyExtractor.extract_from_path('wms\\boc\\tsd\\01\\output.md')
# Returns: {'project': 'wms', 'activity': 'wms\\boc', 'doc_type': 'wms\\boc\\tsd'}

# Get display name
HierarchyExtractor.get_hierarchy_display_name('by_project')  # Returns: 'Project'

# Get all hierarchy labels
HierarchyExtractor.get_hierarchy_labels()
# Returns: [('by_project', 'By Project'), ('by_activity', 'By Activity'), ...]
```

### 2. **HierarchicalMetricsCalculator**
Menghitung dan aggregate metrics

```python
# Calculate metrics untuk satu level
metrics = HierarchicalMetricsCalculator.calculate_metrics_for_level(results_dict)

# Calculate semua levels sekaligus
all_metrics = HierarchicalMetricsCalculator.calculate_all_hierarchies(hierarchical_results)
```

### 3. **HierarchicalMetricsStorage**
Manage storage dan update hierarchical metrics

```python
# Initialize storage
storage = HierarchicalMetricsStorage.initialize_storage()

# Update metrics untuk document
HierarchicalMetricsStorage.update_metrics(
    storage,
    hierarchy_path={'project': 'wms', 'activity': 'wms\\boc', 'doc_type': 'wms\\boc\\tsd'},
    precision=0.95,
    correct_chunks=2,
    positions_found=[1, 3]
)
```

### 4. **HierarchicalHTMLGenerator**
Generate HTML untuk display hierarchical metrics

```python
# Generate table HTML untuk level tertentu
table_html = HierarchicalHTMLGenerator.generate_table_html('by_project', hierarchical_metrics)

# Generate buttons untuk hierarchy selector
buttons_html = HierarchicalHTMLGenerator.generate_hierarchy_buttons()

# Generate content divs
content_divs_html = HierarchicalHTMLGenerator.generate_hierarchy_content_divs(hierarchical_metrics)

# Get CSS styles
css = HierarchicalHTMLGenerator.get_css_styles()

# Get JavaScript functions
js = HierarchicalHTMLGenerator.get_javascript_functions()
```

---

## Perubahan di `evaluator.py`

### Sebelum (Old Code)
```python
# Direct storage initialization
self.hierarchical_results = {
    'by_project': defaultdict(lambda: {...}),
    'by_activity': defaultdict(lambda: {...}),
    'by_doc_type': defaultdict(lambda: {...})
}

# Direct hierarchy extraction
def extract_hierarchy_from_path(self, normalized_path: str):
    parts = normalized_path.split('\\')
    return {...}

# Direct metrics update (30+ lines of repetitive code)
if expected_docs_normalized:
    for doc in expected_docs_normalized:
        hierarchy = self.extract_hierarchy_from_path(doc)
        # Manual update untuk each level...
        self.hierarchical_results['by_project'][project]['total_questions'] += 1
        # ... etc
```

### Sesudah (Refactored)
```python
# Using utility for storage
from utils.hierarchical_metrics import (
    HierarchyExtractor,
    HierarchicalMetricsCalculator,
    HierarchicalMetricsStorage
)

# Initialize using utility
self.hierarchical_results = HierarchicalMetricsStorage.initialize_storage()

# Delegate to utility
def extract_hierarchy_from_path(self, normalized_path: str):
    return HierarchyExtractor.extract_from_path(normalized_path)

# Using utility for update (single line!)
if expected_docs_normalized:
    for doc in expected_docs_normalized:
        hierarchy = self.extract_hierarchy_from_path(doc)
        HierarchicalMetricsStorage.update_metrics(
            self.hierarchical_results,
            hierarchy,
            precision,
            correct_chunks,
            positions_found
        )

# Using utility for calculation
def calculate_hierarchical_metrics(self):
    return HierarchicalMetricsCalculator.calculate_all_hierarchies(
        self.hierarchical_results
    )
```

**Benefits:**
- ✅ Reduced code duplication
- ✅ Easier to maintain
- ✅ More testable
- ✅ Clear separation of concerns

---

## Perubahan di `dashboard_generator.py`

### Sebelum (Old Code)
```python
# 50+ lines untuk generate hierarchical table HTML
def generate_hierarchical_table_html(self, hierarchy_type: str):
    if hierarchy_type not in self.hierarchical_metrics:
        return "<p>No data available</p>"

    data = self.hierarchical_metrics[hierarchy_type]
    rows = []
    for name, metrics in sorted(data.items()):
        success = metrics.get('success_rate', 0)
        precision = metrics.get('average_precision', 0)
        status_class = 'status-good' if success >= 0.8 else (...)
        status_icon = self.get_status_icon(success, 0.8)
        # ... build HTML ...

    return f"<table>...</table>"
```

### Sesudah (Refactored)
```python
from utils.hierarchical_metrics import HierarchicalHTMLGenerator

# Single line delegation!
def generate_hierarchical_table_html(self, hierarchy_type: str):
    return HierarchicalHTMLGenerator.generate_table_html(
        hierarchy_type, self.hierarchical_metrics
    )
```

**Benefits:**
- ✅ 40+ lines of code eliminated
- ✅ HTML generation logic in one place
- ✅ Easier to modify styling/HTML structure
- ✅ Reusable across different dashboards

---

## Import Usage

### Di `evaluator.py`
```python
from utils.hierarchical_metrics import (
    HierarchyExtractor,
    HierarchicalMetricsCalculator,
    HierarchicalMetricsStorage
)
```

### Di `dashboard_generator.py`
```python
from utils.hierarchical_metrics import HierarchicalHTMLGenerator
```

### Di file lain (jika perlu)
```python
from evaluation.utils import (
    HierarchyExtractor,
    HierarchicalMetricsCalculator,
    HierarchicalMetricsStorage,
    HierarchicalHTMLGenerator
)
```

---

## Testing Refactored Code

Jalankan test untuk memastikan refactoring tidak merusak functionality:

```bash
# Run evaluator
python evaluation/evaluator.py

# Generate dashboard
python evaluation/dashboard_generator.py output/evaluation_results_new.json output/dashboard.html

# Run test script
python test_hierarchical_metrics.py
```

Expected output: **Semua metrics dan dashboard berfungsi sama seperti sebelumnya**

---

## Code Quality Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Lines of Code (core logic)** | ~600 | ~400 |
| **Duplication** | High (update logic repeated 3x) | None (single utility method) |
| **Maintainability** | Medium | High |
| **Testability** | Low (tightly coupled) | High (utilities can be unit tested) |
| **Reusability** | Low (hard-coded in evaluator/dashboard) | High (shareable utilities) |

---

## Migration Path

Jika ada file lain yang perlu menggunakan hierarchical metrics:

1. **Import utilities**
   ```python
   from utils.hierarchical_metrics import HierarchyExtractor, ...
   ```

2. **Use utilities instead of reimplementing**
   ```python
   # Instead of custom logic
   hierarchy = HierarchyExtractor.extract_from_path(path)
   ```

3. **No need to modify evaluator/dashboard again**
   - Utilities are self-contained
   - Can be imported and used anywhere in evaluation folder

---

## Files Modified

| File | Changes | Lines Changed |
|------|---------|----------------|
| evaluator.py | Imports + delegate to utils | ~20 lines simplified |
| dashboard_generator.py | Imports + delegate to utils | ~40 lines simplified |
| utils/__init__.py | NEW - Package initialization | - |
| utils/hierarchical_metrics.py | NEW - Core utilities | ~400 lines |

**Net Result: Cleaner code + shared utilities**

---

## Summary

✅ Hierarchical metrics logic terkonsolidasi di utils
✅ Reduced code duplication
✅ Better separation of concerns
✅ Easier to test and maintain
✅ More reusable across the project

**Backward Compatibility: 100% maintained** - Functionality tidak berubah, hanya internal structure yang di-refactor.
