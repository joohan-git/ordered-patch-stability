"""Distinct tilings after primitive-period reduction, translations and D4 action."""
from pathlib import Path
import sys
DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA.parent))
import json
import numpy as np
from computations.common.runtime import output_dir


def canonical(pattern):
    pattern = np.array(pattern, dtype=np.uint8)
    # Remove repeated rows/columns before comparing rectangular unit cells.
    for axis in (0, 1):
        for period in range(1, pattern.shape[axis]+1):
            if pattern.shape[axis] % period == 0 and np.array_equal(pattern, np.roll(pattern, period, axis)):
                pattern = np.take(pattern, range(period), axis=axis)
                break
    images = []
    for rotation in range(4):
        rotated = np.rot90(pattern, rotation) ^ (rotation % 2)
        for mirror in (False, True):
            transformed = np.flip(rotated, 0) ^ 1 if mirror else rotated
            for x in range(transformed.shape[0]):
                for y in range(transformed.shape[1]):
                    images.append((transformed.shape, np.roll(np.roll(transformed, x, 0), y, 1).tobytes()))
    return min(images)


def calculate():
    periodic = json.loads((DATA / 'truchet_2d/periodic.json').read_text())
    rules = json.loads((DATA / 'truchet_2d/rules.json').read_text())
    general, no_checker, sparse = [{canonical(r['pattern']) for r in rows}
                                  for rows in (periodic['runs'], rules['periodic'], rules['strong_periodic'])]
    union = general | no_checker | sparse
    groups = dict(checkerboard_allowed=union - no_checker - sparse,
                  no_checkerboard=no_checker - sparse, sparse_rule=sparse)
    assert sum(map(len, groups.values())) == len(union)
    representatives = {}
    for label, keys in groups.items():
        representatives[label] = [np.frombuffer(bits, dtype=np.uint8).reshape(shape).tolist() for shape, bits in sorted(keys)]
    return dict(sources=['truchet_2d/periodic.json', 'truchet_2d/rules.json'],
                equivalence='Primitive rectangular periods, translations and D4; tile states exchange under quarter turns and axis mirrors.',
                classification='Membership in the sampled cohorts, with sparse-rule priority over no-checkerboard membership. Counts describe the bundled candidate set.',
                total=len(union), counts={label:len(keys) for label,keys in groups.items()}, representatives=representatives)


if __name__ == '__main__':
    result = calculate()
    (output_dir() / 'truchet_2d/tiling_classification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'representatives'}, indent=2))
