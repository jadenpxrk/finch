use super::*;

pub(super) struct DenseMemIndex {
    pub(super) dim: usize,
    metric_type: MetricType,
    quantize: QuantizeType,
    vec_bytes: usize,
    metric: Box<dyn Metric>,
    pos: HashMap<u64, usize>,
    pub(super) keys: Vec<u64>,
    pub(super) vectors: Vec<f32>, // row-major f32 (always present for refiner/exact)
    q_vectors: Vec<u8>, // row-major encoded vectors (only used when quantize != Undefined)
}

impl DenseMemIndex {
    pub(super) fn new(dim: usize, metric_type: MetricType, quantize: QuantizeType) -> Self {
        DenseMemIndex {
            dim,
            metric_type,
            quantize,
            vec_bytes: bytes_per_vector(quantize, dim),
            metric: make_metric(metric_type),
            pos: HashMap::new(),
            keys: Vec::new(),
            vectors: Vec::new(),
            q_vectors: Vec::new(),
        }
    }

    pub(super) fn add(&mut self, key: u64, vector: &[f32]) -> ZResult<()> {
        if vector.len() != self.dim {
            return Err(Status::invalid_argument("vector dim mismatch"));
        }
        if let Some(&i) = self.pos.get(&key) {
            let start = i * self.dim;
            self.vectors[start..start + self.dim].copy_from_slice(vector);
            if self.quantize != QuantizeType::Undefined {
                let mut tmp = Vec::with_capacity(self.vec_bytes);
                quantize_append(self.quantize, vector, &mut tmp);
                let off = i * self.vec_bytes;
                self.q_vectors[off..off + self.vec_bytes].copy_from_slice(&tmp);
            }
            return Ok(());
        }

        let i = self.keys.len();
        self.pos.insert(key, i);
        self.keys.push(key);
        self.vectors.extend_from_slice(vector);
        if self.quantize != QuantizeType::Undefined {
            quantize_append(self.quantize, vector, &mut self.q_vectors);
        }
        Ok(())
    }

    pub(super) fn remove(&mut self, key: u64) {
        let Some(i) = self.pos.remove(&key) else {
            return;
        };
        let last = match self.keys.len() {
            0 => return,
            n => n - 1,
        };

        if i != last {
            let moved_key = self.keys[last];
            self.keys[i] = moved_key;
            self.pos.insert(moved_key, i);

            // Swap last row into i.
            let src = last * self.dim;
            let dst = i * self.dim;
            self.vectors.copy_within(src..src + self.dim, dst);
            if self.quantize != QuantizeType::Undefined {
                let srcb = last * self.vec_bytes;
                let dstb = i * self.vec_bytes;
                self.q_vectors
                    .copy_within(srcb..srcb + self.vec_bytes, dstb);
            }
        }

        self.keys.pop();
        self.vectors.truncate(self.keys.len() * self.dim);
        if self.quantize != QuantizeType::Undefined {
            self.q_vectors.truncate(self.keys.len() * self.vec_bytes);
        }
    }

    pub(super) fn search(
        &self,
        query: &[f32],
        topk: usize,
        mut allow: impl FnMut(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if self.keys.is_empty() || topk == 0 {
            return Vec::new();
        }
        let mut heap = TopkHeap::new(topk);

        if self.quantize == QuantizeType::Undefined {
            for (i, &key) in self.keys.iter().enumerate() {
                if !allow(key) {
                    continue;
                }
                let start = i * self.dim;
                let v = &self.vectors[start..start + self.dim];
                let d = self.metric.distance(query, v);
                heap.push(d, key);
            }
        } else {
            for (i, &key) in self.keys.iter().enumerate() {
                if !allow(key) {
                    continue;
                }
                let off = i * self.vec_bytes;
                let data = &self.q_vectors[off..off + self.vec_bytes];
                let d =
                    distance_to_quantized(self.metric_type, query, data, self.dim, self.quantize);
                heap.push(d, key);
            }
        }
        heap.into_sorted()
    }

    pub(super) fn search_by_keys(
        &self,
        query: &[f32],
        topk: usize,
        keys: impl IntoIterator<Item = u64>,
        mut allow: impl FnMut(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if self.keys.is_empty() || topk == 0 {
            return Vec::new();
        }
        let mut heap = TopkHeap::new(topk);

        if self.quantize == QuantizeType::Undefined {
            for key in keys {
                if !allow(key) {
                    continue;
                }
                let Some(&i) = self.pos.get(&key) else {
                    continue;
                };
                let start = i * self.dim;
                let v = &self.vectors[start..start + self.dim];
                let d = self.metric.distance(query, v);
                heap.push(d, key);
            }
        } else {
            for key in keys {
                if !allow(key) {
                    continue;
                }
                let Some(&i) = self.pos.get(&key) else {
                    continue;
                };
                let off = i * self.vec_bytes;
                let data = &self.q_vectors[off..off + self.vec_bytes];
                let d =
                    distance_to_quantized(self.metric_type, query, data, self.dim, self.quantize);
                heap.push(d, key);
            }
        }

        heap.into_sorted()
    }
}

pub(super) struct Binary32MemIndex {
    dim: usize,
    pos: HashMap<u64, usize>,
    keys: Vec<u64>,
    vectors: Vec<u32>, // row-major u32 words
}

impl Binary32MemIndex {
    pub(super) fn new(dim: usize) -> Self {
        Binary32MemIndex {
            dim,
            pos: HashMap::new(),
            keys: Vec::new(),
            vectors: Vec::new(),
        }
    }

    pub(super) fn add(&mut self, key: u64, vector: &[u32]) -> ZResult<()> {
        if vector.len() != self.dim {
            return Err(Status::invalid_argument("vector dim mismatch"));
        }
        if let Some(&i) = self.pos.get(&key) {
            let start = i * self.dim;
            self.vectors[start..start + self.dim].copy_from_slice(vector);
            return Ok(());
        }
        let i = self.keys.len();
        self.pos.insert(key, i);
        self.keys.push(key);
        self.vectors.extend_from_slice(vector);
        Ok(())
    }

    pub(super) fn remove(&mut self, key: u64) {
        let Some(i) = self.pos.remove(&key) else {
            return;
        };
        let last = match self.keys.len() {
            0 => return,
            n => n - 1,
        };

        if i != last {
            let moved_key = self.keys[last];
            self.keys[i] = moved_key;
            self.pos.insert(moved_key, i);

            let src = last * self.dim;
            let dst = i * self.dim;
            self.vectors.copy_within(src..src + self.dim, dst);
        }

        self.keys.pop();
        self.vectors.truncate(self.keys.len() * self.dim);
    }

    pub(super) fn search(
        &self,
        query: &[u32],
        topk: usize,
        mut allow: impl FnMut(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if self.keys.is_empty() || topk == 0 {
            return Vec::new();
        }
        let mut heap = TopkHeap::new(topk);
        for (i, &key) in self.keys.iter().enumerate() {
            if !allow(key) {
                continue;
            }
            let start = i * self.dim;
            let v = &self.vectors[start..start + self.dim];
            let mut acc: u32 = 0;
            for j in 0..self.dim.min(query.len()) {
                acc = acc.wrapping_add((v[j] ^ query[j]).count_ones());
            }
            heap.push(acc as f32, key);
        }
        heap.into_sorted()
    }

    pub(super) fn search_by_keys(
        &self,
        query: &[u32],
        topk: usize,
        keys: impl IntoIterator<Item = u64>,
        mut allow: impl FnMut(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if self.keys.is_empty() || topk == 0 {
            return Vec::new();
        }
        let mut heap = TopkHeap::new(topk);
        for key in keys {
            if !allow(key) {
                continue;
            }
            let Some(&i) = self.pos.get(&key) else {
                continue;
            };
            let start = i * self.dim;
            let v = &self.vectors[start..start + self.dim];
            let mut acc: u32 = 0;
            for j in 0..self.dim.min(query.len()) {
                acc = acc.wrapping_add((v[j] ^ query[j]).count_ones());
            }
            heap.push(acc as f32, key);
        }
        heap.into_sorted()
    }
}

pub(super) struct Binary64MemIndex {
    dim: usize,
    pos: HashMap<u64, usize>,
    keys: Vec<u64>,
    vectors: Vec<u64>, // row-major u64 words
}

impl Binary64MemIndex {
    pub(super) fn new(dim: usize) -> Self {
        Binary64MemIndex {
            dim,
            pos: HashMap::new(),
            keys: Vec::new(),
            vectors: Vec::new(),
        }
    }

    pub(super) fn add(&mut self, key: u64, vector: &[u64]) -> ZResult<()> {
        if vector.len() != self.dim {
            return Err(Status::invalid_argument("vector dim mismatch"));
        }
        if let Some(&i) = self.pos.get(&key) {
            let start = i * self.dim;
            self.vectors[start..start + self.dim].copy_from_slice(vector);
            return Ok(());
        }
        let i = self.keys.len();
        self.pos.insert(key, i);
        self.keys.push(key);
        self.vectors.extend_from_slice(vector);
        Ok(())
    }

    pub(super) fn remove(&mut self, key: u64) {
        let Some(i) = self.pos.remove(&key) else {
            return;
        };
        let last = match self.keys.len() {
            0 => return,
            n => n - 1,
        };

        if i != last {
            let moved_key = self.keys[last];
            self.keys[i] = moved_key;
            self.pos.insert(moved_key, i);

            let src = last * self.dim;
            let dst = i * self.dim;
            self.vectors.copy_within(src..src + self.dim, dst);
        }

        self.keys.pop();
        self.vectors.truncate(self.keys.len() * self.dim);
    }

    pub(super) fn search(
        &self,
        query: &[u64],
        topk: usize,
        mut allow: impl FnMut(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if self.keys.is_empty() || topk == 0 {
            return Vec::new();
        }
        let mut heap = TopkHeap::new(topk);
        for (i, &key) in self.keys.iter().enumerate() {
            if !allow(key) {
                continue;
            }
            let start = i * self.dim;
            let v = &self.vectors[start..start + self.dim];
            let mut acc: u32 = 0;
            for j in 0..self.dim.min(query.len()) {
                acc = acc.wrapping_add((v[j] ^ query[j]).count_ones());
            }
            heap.push(acc as f32, key);
        }
        heap.into_sorted()
    }

    pub(super) fn search_by_keys(
        &self,
        query: &[u64],
        topk: usize,
        keys: impl IntoIterator<Item = u64>,
        mut allow: impl FnMut(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if self.keys.is_empty() || topk == 0 {
            return Vec::new();
        }
        let mut heap = TopkHeap::new(topk);
        for key in keys {
            if !allow(key) {
                continue;
            }
            let Some(&i) = self.pos.get(&key) else {
                continue;
            };
            let start = i * self.dim;
            let v = &self.vectors[start..start + self.dim];
            let mut acc: u32 = 0;
            for j in 0..self.dim.min(query.len()) {
                acc = acc.wrapping_add((v[j] ^ query[j]).count_ones());
            }
            heap.push(acc as f32, key);
        }
        heap.into_sorted()
    }
}

pub(super) struct SparseMemIndex {
    quantize: QuantizeType,
    pos: HashMap<u64, usize>,
    keys: Vec<u64>,
    vectors: Vec<SparseVector>,
}

impl SparseMemIndex {
    pub(super) fn new(quantize: QuantizeType) -> Self {
        SparseMemIndex {
            quantize,
            pos: HashMap::new(),
            keys: Vec::new(),
            vectors: Vec::new(),
        }
    }

    pub(super) fn add(&mut self, key: u64, vector: &SparseVector) -> ZResult<()> {
        let vec = match self.quantize {
            QuantizeType::Undefined => vector.clone(),
            QuantizeType::Fp16 => vector.quantize_to_f16(),
            _ => {
                return Err(Status::invalid_argument(
                    "sparse quantize only supports Fp16",
                ));
            }
        };
        if let Some(&i) = self.pos.get(&key) {
            self.vectors[i] = vec;
            return Ok(());
        }
        let i = self.keys.len();
        self.pos.insert(key, i);
        self.keys.push(key);
        self.vectors.push(vec);
        Ok(())
    }

    pub(super) fn remove(&mut self, key: u64) {
        let Some(i) = self.pos.remove(&key) else {
            return;
        };
        let last = match self.keys.len() {
            0 => return,
            n => n - 1,
        };
        if i != last {
            let moved_key = self.keys[last];
            self.keys[i] = moved_key;
            self.pos.insert(moved_key, i);
            self.vectors[i] = self.vectors[last].clone();
        }
        self.keys.pop();
        self.vectors.pop();
    }

    pub(super) fn search(
        &self,
        query: &SparseVector,
        topk: usize,
        mut allow: impl FnMut(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if self.keys.is_empty() || topk == 0 {
            return Vec::new();
        }
        let mut heap = TopkHeap::new(topk);
        for (i, &key) in self.keys.iter().enumerate() {
            if !allow(key) {
                continue;
            }
            // Neg inner product: smaller = more similar
            let d = -query.dot(&self.vectors[i]);
            heap.push(d, key);
        }
        heap.into_sorted()
    }

    pub(super) fn search_by_keys(
        &self,
        query: &SparseVector,
        topk: usize,
        keys: impl IntoIterator<Item = u64>,
        mut allow: impl FnMut(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if self.keys.is_empty() || topk == 0 {
            return Vec::new();
        }
        let mut heap = TopkHeap::new(topk);
        for key in keys {
            if !allow(key) {
                continue;
            }
            let Some(&i) = self.pos.get(&key) else {
                continue;
            };
            let d = -query.dot(&self.vectors[i]);
            heap.push(d, key);
        }
        heap.into_sorted()
    }
}
