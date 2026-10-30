// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Immutable, sparse page-index snapshots keyed by global row group and physical leaf.

use parquet::file::metadata::page_index::{PageIndex, PageIndexProvider};
use parquet::file::metadata::{ColumnChunkMask, ParquetMetaData};
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::page_index::offset_index::OffsetIndexMetaData;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

/// Independent column- and offset-index requests in file-global coordinates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageIndexSelection {
    column_indexes: HashSet<(usize, usize)>,
    offset_indexes: HashSet<(usize, usize)>,
}

impl PageIndexSelection {
    /// Creates an empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests a column index for a global row group and physical leaf.
    pub fn add_column_index(&mut self, row_group: usize, physical_leaf: usize) {
        self.column_indexes.insert((row_group, physical_leaf));
    }

    /// Requests an offset index for a global row group and physical leaf.
    pub fn add_offset_index(&mut self, row_group: usize, physical_leaf: usize) {
        self.offset_indexes.insert((row_group, physical_leaf));
    }

    /// Returns the requested column-index coordinates.
    pub fn column_indexes(&self) -> &HashSet<(usize, usize)> {
        &self.column_indexes
    }

    /// Returns the requested offset-index coordinates.
    pub fn offset_indexes(&self) -> &HashSet<(usize, usize)> {
        &self.offset_indexes
    }

    /// Returns whether neither kind of index is requested.
    pub fn is_empty(&self) -> bool {
        self.column_indexes.is_empty() && self.offset_indexes.is_empty()
    }

    /// Returns the column-index mask, covering the Cartesian product of its axes.
    pub fn column_index_mask(&self) -> ColumnChunkMask {
        Self::mask(&self.column_indexes)
    }

    /// Returns the offset-index mask, covering the Cartesian product of its axes.
    pub fn offset_index_mask(&self) -> ColumnChunkMask {
        Self::mask(&self.offset_indexes)
    }

    fn mask(entries: &HashSet<(usize, usize)>) -> ColumnChunkMask {
        ColumnChunkMask::row_groups_and_columns(
            entries.iter().map(|&(row_group, _)| row_group),
            entries.iter().map(|&(_, leaf)| leaf),
        )
    }

    /// Returns unresolved requests whose footer advertises an index range.
    ///
    /// A generic provider's `None` means unknown, not known absent. Only this
    /// cache's explicitly resolved negative entries suppress such requests.
    /// Coordinates must be valid global row groups and physical leaves in `metadata`.
    pub fn missing(
        &self,
        metadata: &ParquetMetaData,
        provider: Option<&dyn PageIndexProvider>,
    ) -> Self {
        let cached =
            provider.and_then(|p| p.as_any().downcast_ref::<CachedPageIndexes>());
        let mut missing = Self::new();
        for &(rg, leaf) in &self.column_indexes {
            if metadata
                .row_group(rg)
                .column(leaf)
                .column_index_range()
                .is_some()
                && !cached.is_some_and(|c| c.column_indexes.contains_key(&(rg, leaf)))
                && provider.is_none_or(|p| p.column_index(rg, leaf).is_none())
            {
                missing.add_column_index(rg, leaf);
            }
        }
        for &(rg, leaf) in &self.offset_indexes {
            if metadata
                .row_group(rg)
                .column(leaf)
                .offset_index_range()
                .is_some()
                && !cached.is_some_and(|c| c.offset_indexes.contains_key(&(rg, leaf)))
                && provider.is_none_or(|p| p.offset_index(rg, leaf).is_none())
            {
                missing.add_offset_index(rg, leaf);
            }
        }
        missing
    }
}

/// Decoded indexes and explicit negative results, shared across immutable snapshots.
///
/// An absent map key is unresolved; a key mapped to `None` was decoded but absent.
#[derive(Debug, Clone, Default)]
pub struct CachedPageIndexes {
    column_indexes: HashMap<(usize, usize), Option<Arc<ColumnIndexMetaData>>>,
    offset_indexes: HashMap<(usize, usize), Option<Arc<OffsetIndexMetaData>>>,
}

impl CachedPageIndexes {
    /// Moves decoded metadata into Arcs without copying index payloads.
    ///
    /// Call only after successfully decoding both masks of `loaded_selection`.
    /// All coordinates in each mask's Cartesian product are resolved, including
    /// those added by widening a non-rectangular selection. Coordinates must be
    /// valid global row groups and physical leaves for the decoded file.
    pub fn from_decoded(
        page_index: Option<PageIndex>,
        loaded_selection: &PageIndexSelection,
    ) -> Self {
        fn resolved<T>(
            entries: &HashSet<(usize, usize)>,
        ) -> HashMap<(usize, usize), Option<Arc<T>>> {
            let row_groups: HashSet<_> = entries.iter().map(|&(rg, _)| rg).collect();
            let leaves: HashSet<_> = entries.iter().map(|&(_, leaf)| leaf).collect();
            row_groups
                .into_iter()
                .flat_map(|rg| leaves.iter().map(move |&leaf| ((rg, leaf), None)))
                .collect()
        }
        let mut result = Self {
            column_indexes: resolved(&loaded_selection.column_indexes),
            offset_indexes: resolved(&loaded_selection.offset_indexes),
        };
        if let Some(page_index) = page_index {
            let (columns, offsets) = page_index.into_index_entries();
            for (key, index) in columns {
                if let Some(entry) = result.column_indexes.get_mut(&key) {
                    *entry = Some(Arc::new(index));
                }
            }
            for (key, index) in offsets {
                if let Some(entry) = result.offset_indexes.get_mut(&key) {
                    *entry = Some(Arc::new(index));
                }
            }
        }
        result
    }

    /// Copies selected available indexes from a provider, sharing this cache's Arcs.
    /// Generic providers do not expose resolved coverage, so their `None` is omitted.
    pub fn from_provider(
        provider: &dyn PageIndexProvider,
        selection: &PageIndexSelection,
    ) -> Self {
        if let Some(cached) = provider.as_any().downcast_ref::<Self>() {
            return cached.subset(selection);
        }
        Self {
            column_indexes: selection
                .column_indexes
                .iter()
                .filter_map(|&(rg, leaf)| {
                    provider
                        .column_index(rg, leaf)
                        .map(|index| ((rg, leaf), Some(Arc::new(index.clone()))))
                })
                .collect(),
            offset_indexes: selection
                .offset_indexes
                .iter()
                .filter_map(|&(rg, leaf)| {
                    provider
                        .offset_index(rg, leaf)
                        .map(|index| ((rg, leaf), Some(Arc::new(index.clone()))))
                })
                .collect(),
        }
    }

    /// Returns a union without modifying either snapshot, preferring positive results.
    /// If both snapshots have a positive result, retains this snapshot's Arc.
    pub fn merge(&self, other: &Self) -> Self {
        fn merge<T>(
            left: &mut HashMap<(usize, usize), Option<Arc<T>>>,
            right: &HashMap<(usize, usize), Option<Arc<T>>>,
        ) {
            for (&key, value) in right {
                let entry = left.entry(key).or_default();
                if entry.is_none() {
                    entry.clone_from(value);
                }
            }
        }
        let mut result = self.clone();
        merge(&mut result.column_indexes, &other.column_indexes);
        merge(&mut result.offset_indexes, &other.offset_indexes);
        result
    }

    /// Returns precisely the selected resolved entries, sharing their payloads.
    pub fn subset(&self, selection: &PageIndexSelection) -> Self {
        Self {
            column_indexes: selection
                .column_indexes
                .iter()
                .filter_map(|key| {
                    self.column_indexes
                        .get(key)
                        .map(|value| (*key, value.clone()))
                })
                .collect(),
            offset_indexes: selection
                .offset_indexes
                .iter()
                .filter_map(|key| {
                    self.offset_indexes
                        .get(key)
                        .map(|value| (*key, value.clone()))
                })
                .collect(),
        }
    }

    /// Estimates retained bytes, charging shared payloads fully to each snapshot.
    ///
    /// Mirrors Parquet's private `metadata::memory::HeapSize` layout: inline
    /// structures, Arc counters, map buckets, index vectors and binary bounds.
    /// Public Vec capacities are used where available. Column-index accessors
    /// expose only slices for bounds, null pages and histograms, so their logical
    /// lengths are charged; spare capacity is inaccessible. Map bucket overhead
    /// follows Parquet's hashbrown approximation, not an allocator-exact bound.
    pub fn memory_size(&self) -> usize {
        size_of::<Self>()
            + map_size(&self.column_indexes)
            + map_size(&self.offset_indexes)
            + self
                .column_indexes
                .values()
                .flatten()
                .map(|index| {
                    2 * size_of::<usize>()
                        + size_of::<ColumnIndexMetaData>()
                        + column_index_heap_size(index)
                })
                .sum::<usize>()
            + self
                .offset_indexes
                .values()
                .flatten()
                .map(|index| {
                    2 * size_of::<usize>()
                        + size_of::<OffsetIndexMetaData>()
                        + index.page_locations().capacity()
                            * size_of::<
                                parquet::file::page_index::offset_index::PageLocation,
                            >()
                        + index
                            .unencoded_byte_array_data_bytes()
                            .map_or(0, |values| values.capacity() * size_of::<i64>())
                })
                .sum::<usize>()
    }
}

impl PageIndexProvider for CachedPageIndexes {
    fn has_column_indexes(&self) -> bool {
        self.column_indexes.values().any(Option::is_some)
    }

    fn has_offset_indexes(&self) -> bool {
        self.offset_indexes.values().any(Option::is_some)
    }

    fn column_index(
        &self,
        row_group: usize,
        physical_leaf: usize,
    ) -> Option<&ColumnIndexMetaData> {
        self.column_indexes
            .get(&(row_group, physical_leaf))?
            .as_deref()
    }

    fn offset_index(
        &self,
        row_group: usize,
        physical_leaf: usize,
    ) -> Option<&OffsetIndexMetaData> {
        self.offset_indexes
            .get(&(row_group, physical_leaf))?
            .as_deref()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn map_size<K, V>(map: &HashMap<K, V>) -> usize {
    let capacity = map.capacity();
    if capacity == 0 {
        return 0;
    }
    // These maps' key/value pairs exceed three bytes, matching Parquet HeapSize's
    // four-bucket minimum and 7/8 load-factor model.
    let buckets = if capacity < 4 {
        4
    } else if capacity < 8 {
        8
    } else if capacity < 15 {
        16
    } else {
        (capacity.saturating_mul(8) / 7).next_power_of_two()
    };
    16 + buckets * (size_of::<(K, V)>() + 1)
}

fn column_index_heap_size(index: &ColumnIndexMetaData) -> usize {
    let pages = index.num_pages() as usize;
    let common = pages * size_of::<bool>()
        + index
            .null_counts()
            .map_or(0, |values| values.capacity() * size_of::<i64>())
        + index
            .nan_counts()
            .map_or(0, |values| values.capacity() * size_of::<i64>())
        + (0..pages)
            .map(|page| {
                index
                    .repetition_level_histogram(page)
                    .map_or(0, size_of_val)
                    + index
                        .definition_level_histogram(page)
                        .map_or(0, size_of_val)
            })
            .sum::<usize>();
    let bounds = match index {
        ColumnIndexMetaData::BOOLEAN(i) => {
            size_of_val(i.min_values()) + size_of_val(i.max_values())
        }
        ColumnIndexMetaData::INT32(i) => {
            size_of_val(i.min_values()) + size_of_val(i.max_values())
        }
        ColumnIndexMetaData::INT64(i) => {
            size_of_val(i.min_values()) + size_of_val(i.max_values())
        }
        ColumnIndexMetaData::INT96(i) => {
            size_of_val(i.min_values()) + size_of_val(i.max_values())
        }
        ColumnIndexMetaData::FLOAT(i) => {
            size_of_val(i.min_values()) + size_of_val(i.max_values())
        }
        ColumnIndexMetaData::DOUBLE(i) => {
            size_of_val(i.min_values()) + size_of_val(i.max_values())
        }
        ColumnIndexMetaData::BYTE_ARRAY(i)
        | ColumnIndexMetaData::FIXED_LEN_BYTE_ARRAY(i) => {
            i.min_values_iter()
                .flatten()
                .map(<[u8]>::len)
                .sum::<usize>()
                + i.max_values_iter()
                    .flatten()
                    .map(<[u8]>::len)
                    .sum::<usize>()
                + 2 * (pages + 1) * size_of::<usize>()
        }
    };
    common + bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::basic::Type as PhysicalType;
    use parquet::file::metadata::page_index::PageIndexBuilder;
    use parquet::file::metadata::{
        ColumnChunkMetaData, ColumnIndexBuilder, FileMetaData, LevelHistogram,
        OffsetIndexBuilder, RowGroupMetaData,
    };
    use parquet::schema::types::{SchemaDescriptor, Type};

    fn metadata(with_ranges: bool) -> ParquetMetaData {
        let fields = (0..3)
            .map(|leaf| {
                Arc::new(
                    Type::primitive_type_builder(
                        &format!("c{leaf}"),
                        PhysicalType::INT32,
                    )
                    .build()
                    .unwrap(),
                )
            })
            .collect();
        let schema = Arc::new(SchemaDescriptor::new(Arc::new(
            Type::group_type_builder("schema")
                .with_fields(fields)
                .build()
                .unwrap(),
        )));
        let row_groups = (0..3)
            .map(|_| {
                let columns = (0..3)
                    .map(|leaf| {
                        let builder = ColumnChunkMetaData::builder(schema.column(leaf));
                        let builder = if with_ranges {
                            builder
                                .set_column_index_offset(Some(100))
                                .set_column_index_length(Some(20))
                                .set_offset_index_offset(Some(120))
                                .set_offset_index_length(Some(20))
                        } else {
                            builder
                        };
                        builder.build().unwrap()
                    })
                    .collect();
                RowGroupMetaData::builder(Arc::clone(&schema))
                    .set_num_rows(1)
                    .set_column_metadata(columns)
                    .build()
                    .unwrap()
            })
            .collect();
        ParquetMetaData::new(
            FileMetaData::new(1, 3, None, None, schema, None),
            row_groups,
        )
    }

    fn column_index() -> ColumnIndexMetaData {
        let mut builder = ColumnIndexBuilder::new(PhysicalType::INT32);
        builder.append(
            false,
            1i32.to_le_bytes().to_vec(),
            2i32.to_le_bytes().to_vec(),
            0,
            None,
        );
        builder.build().unwrap()
    }

    fn offset_index() -> OffsetIndexMetaData {
        let mut builder = OffsetIndexBuilder::new();
        builder.append_row_count(1);
        builder.append_offset_and_size(10, 20);
        builder.build()
    }

    #[test]
    fn page_index_selection_masks_are_independent_cartesian_products() {
        let mut selection = PageIndexSelection::new();
        assert!(selection.is_empty());
        assert!(selection.column_index_mask().is_none());
        assert!(selection.offset_index_mask().is_none());
        selection.add_column_index(0, 0);
        selection.add_column_index(2, 2);
        selection.add_column_index(0, 0);
        selection.add_offset_index(1, 2);
        assert_eq!(selection.column_indexes().len(), 2);
        assert_eq!(
            selection
                .column_index_mask()
                .column_chunk_indices(3, 3)
                .collect::<Vec<_>>(),
            vec![(0, 0), (0, 2), (2, 0), (2, 2)]
        );
        assert_eq!(
            selection
                .offset_index_mask()
                .column_chunk_indices(3, 3)
                .collect::<Vec<_>>(),
            vec![(1, 2)]
        );
        assert_eq!(selection.missing(&metadata(true), None), selection);
        assert!(selection.missing(&metadata(false), None).is_empty());
    }

    #[test]
    fn page_index_selection_missing_distinguishes_generic_none_from_resolved_absence() {
        let mut selection = PageIndexSelection::new();
        selection.add_column_index(0, 0);
        selection.add_column_index(0, 1);
        selection.add_offset_index(0, 0);
        let mut builder = PageIndexBuilder::new(3, 3);
        builder.put_column_index(column_index(), 0, 0);
        let provider = builder.build();
        let missing = selection.missing(&metadata(true), Some(&provider));
        assert_eq!(missing.column_indexes(), &HashSet::from([(0, 1)]));
        assert_eq!(missing.offset_indexes(), &HashSet::from([(0, 0)]));

        let fallback = CachedPageIndexes::from_provider(&provider, &selection);
        assert_eq!(selection.missing(&metadata(true), Some(&fallback)), missing);
        let decoded = CachedPageIndexes::from_decoded(Some(provider), &selection);
        assert!(
            selection
                .missing(&metadata(true), Some(&decoded))
                .is_empty()
        );
        assert!(decoded.has_column_indexes());
        assert!(!decoded.has_offset_indexes());
        assert!(!decoded.is_complete());
        assert!(decoded.column_index(0, 1).is_none());
        assert!(decoded.offset_index(0, 0).is_none());
    }

    #[test]
    fn cached_page_indexes_from_decoded_resolves_only_loaded_masks_and_moves_payloads() {
        let column = column_index();
        let values_ptr = match &column {
            ColumnIndexMetaData::INT32(index) => index.min_values().as_ptr(),
            _ => unreachable!(),
        };
        let offset = offset_index();
        let locations_ptr = offset.page_locations().as_ptr();
        let mut builder = PageIndexBuilder::new(3, 3);
        builder.put_column_index(column, 2, 0);
        builder.put_offset_index(offset, 1, 2);
        // Not decoded under either requested mask: must not become cached coverage.
        builder.put_column_index(column_index(), 1, 1);
        let mut selection = PageIndexSelection::new();
        selection.add_column_index(0, 0);
        selection.add_column_index(2, 2);
        selection.add_offset_index(1, 2);
        let cached = CachedPageIndexes::from_decoded(Some(builder.build()), &selection);
        assert_eq!(cached.column_indexes.len(), 4);
        assert_eq!(cached.offset_indexes.len(), 1);
        assert!(cached.column_indexes[&(0, 2)].is_none());
        assert!(!cached.column_indexes.contains_key(&(1, 1)));
        assert!(!cached.offset_indexes.contains_key(&(2, 2)));
        match cached.column_index(2, 0).unwrap() {
            ColumnIndexMetaData::INT32(index) => {
                assert_eq!(index.min_values().as_ptr(), values_ptr)
            }
            _ => unreachable!(),
        }
        assert_eq!(
            cached.offset_index(1, 2).unwrap().page_locations().as_ptr(),
            locations_ptr
        );
        assert!(cached.is_complete());
        assert!(selection.missing(&metadata(true), Some(&cached)).is_empty());
    }

    #[test]
    fn cached_page_indexes_merge_is_immutable_positive_union_with_shared_arcs() {
        let mut selection = PageIndexSelection::new();
        selection.add_column_index(0, 0);
        selection.add_offset_index(2, 1);
        let negative = CachedPageIndexes::from_decoded(None, &selection);
        let negative_size = negative.memory_size();
        let mut builder = PageIndexBuilder::new(3, 3);
        builder.put_column_index(column_index(), 0, 0);
        builder.put_offset_index(offset_index(), 2, 1);
        let positive = CachedPageIndexes::from_decoded(Some(builder.build()), &selection);
        let positive_size = positive.memory_size();
        let mut extra_selection = PageIndexSelection::new();
        extra_selection.add_column_index(1, 2);
        let extra = CachedPageIndexes::from_decoded(None, &extra_selection);
        for merged in [
            negative.merge(&positive).merge(&extra),
            positive.merge(&negative).merge(&extra),
        ] {
            assert!(Arc::ptr_eq(
                merged.column_indexes[&(0, 0)].as_ref().unwrap(),
                positive.column_indexes[&(0, 0)].as_ref().unwrap()
            ));
            assert!(Arc::ptr_eq(
                merged.offset_indexes[&(2, 1)].as_ref().unwrap(),
                positive.offset_indexes[&(2, 1)].as_ref().unwrap()
            ));
            assert!(merged.column_indexes.contains_key(&(1, 2)));
            assert!(merged.column_indexes[&(1, 2)].is_none());
            let subset = CachedPageIndexes::from_provider(&merged, &selection);
            assert_eq!(subset.column_indexes.len(), 1);
            assert_eq!(subset.offset_indexes.len(), 1);
            assert!(Arc::ptr_eq(
                subset.column_indexes[&(0, 0)].as_ref().unwrap(),
                positive.column_indexes[&(0, 0)].as_ref().unwrap()
            ));
        }
        assert!(negative.column_indexes[&(0, 0)].is_none());
        assert_eq!(negative.memory_size(), negative_size);
        assert_eq!(positive.memory_size(), positive_size);
        assert!(!positive.column_indexes.contains_key(&(1, 2)));
    }

    #[test]
    fn cached_page_indexes_negative_subset_preserves_resolution_not_availability() {
        let mut selection = PageIndexSelection::new();
        selection.add_column_index(2, 1);
        selection.add_offset_index(0, 2);
        let negative = CachedPageIndexes::from_decoded(None, &selection);
        let subset = CachedPageIndexes::from_provider(&negative, &selection);
        assert!(selection.missing(&metadata(true), Some(&subset)).is_empty());
        assert!(!subset.has_column_indexes());
        assert!(!subset.has_offset_indexes());
        assert!(subset.memory_size() > CachedPageIndexes::default().memory_size());
        assert_eq!(
            subset.subset(&PageIndexSelection::new()).memory_size(),
            size_of::<CachedPageIndexes>()
        );
    }

    #[test]
    fn cached_page_indexes_memory_size_charges_large_binary_bounds_and_histograms() {
        let mut builder = ColumnIndexBuilder::new(PhysicalType::BYTE_ARRAY);
        let bytes_per_bound = 16 * 1024;
        builder.append(
            false,
            vec![b'a'; bytes_per_bound],
            vec![b'z'; bytes_per_bound],
            0,
            None,
        );
        builder.append_histograms(
            &Some(LevelHistogram::from(vec![1, 2, 3])),
            &Some(LevelHistogram::from(vec![4, 5])),
        );
        let index = builder.build().unwrap();
        let expected_heap = 2 * bytes_per_bound
            + 4 * size_of::<usize>()
            + size_of::<bool>()
            + index.null_counts().unwrap().capacity() * size_of::<i64>()
            + 5 * size_of::<i64>();
        assert_eq!(column_index_heap_size(&index), expected_heap);
        let mut builder = PageIndexBuilder::new(3, 3);
        builder.put_column_index(index, 0, 1);
        let mut selection = PageIndexSelection::new();
        selection.add_column_index(0, 1);
        let cached = CachedPageIndexes::from_decoded(Some(builder.build()), &selection);
        assert!(cached.memory_size() > 2 * bytes_per_bound);
        assert_eq!(cached.memory_size(), cached.clone().memory_size());
    }

    #[test]
    fn cached_page_indexes_memory_size_charges_offset_vector_capacities() {
        let mut locations = Vec::with_capacity(32);
        locations.extend(offset_index().page_locations().iter().cloned());
        let mut byte_sizes = Vec::with_capacity(64);
        byte_sizes.push(100);
        let index = OffsetIndexMetaData {
            page_locations: locations,
            unencoded_byte_array_data_bytes: Some(byte_sizes),
        };
        let mut builder = PageIndexBuilder::new(3, 3);
        builder.put_offset_index(index, 0, 0);
        let mut selection = PageIndexSelection::new();
        selection.add_offset_index(0, 0);
        let cached = CachedPageIndexes::from_decoded(Some(builder.build()), &selection);
        let index = cached.offset_index(0, 0).unwrap();
        let expected = size_of::<CachedPageIndexes>()
            + map_size(&cached.offset_indexes)
            + 2 * size_of::<usize>()
            + size_of::<OffsetIndexMetaData>()
            + index.page_locations().capacity()
                * size_of::<parquet::file::page_index::offset_index::PageLocation>()
            + index.unencoded_byte_array_data_bytes().unwrap().capacity()
                * size_of::<i64>();
        assert_eq!(cached.memory_size(), expected);
    }
}
