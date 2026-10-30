// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! End-to-end SQL benchmark for a wide Parquet file's page-index metadata cache.
//!
//! Run separately with PAGE_INDEX_BENCH_COLUMNS=64 and =256. Other dimensions are
//! PAGE_INDEX_BENCH_ROWS (32768), PAGE_INDEX_BENCH_ROW_GROUP_ROWS (4096),
//! PAGE_INDEX_BENCH_PAGE_ROWS (64), and PAGE_INDEX_BENCH_VALUE_BYTES (128).
//! PAGE_INDEX_BENCH_FILE optionally names a persistent fixture: an existing file
//! is validated, never overwritten. PAGE_INDEX_BENCH_REPORT optionally saves an
//! untimed, correctness-checked observation for every case as JSON. For example:
//!
//! ```text
//! PAGE_INDEX_BENCH_FILE=wide.parquet PAGE_INDEX_BENCH_REPORT=selective.json \
//!   cargo bench -p datafusion --bench parquet_page_index_cache -- --test
//! ```
//!
//! Copy this file and its Cargo bench registration to either c9fb12bdd (repair
//! only) or c264d5653 (selective caching), retaining the same Arrow #11475 pin.
//! Fixture generation, registration, cache reset/footer seeding, and warm seeding
//! are excluded from timings. Timings include SQL planning and execution/collect,
//! not result validation or metric snapshots. "Cold" means page-index-cache
//! cold, NOT filesystem/OS cold; footer, statistics and listing caches stay warm.
//! Overlap resets and seeds projection A before EVERY iteration of projection B,
//! so it does not turn into an identical-projection hit after the first sample.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_datasource_parquet::metadata::{CachedParquetMetaData, DFParquetMetadata};
use datafusion_physical_plan::metrics::MetricValue;
use datafusion_physical_plan::{ExecutionPlan, collect};
use object_store::ObjectStoreExt;
use object_store::local::LocalFileSystem;
use parquet::arrow::{ArrowWriter, parquet_to_arrow_schema};
use parquet::basic::Compression;
use parquet::file::metadata::{KeyValue, PageIndexPolicy, ParquetMetaDataReader};
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use serde_json::{Value, json};
use tokio::runtime::Runtime;

const FIXTURE_KEY: &str = "datafusion.page_index_cache_bench.v1";

struct Dimensions {
    columns: usize,
    rows: usize,
    row_group_rows: usize,
    page_rows: usize,
    value_bytes: usize,
}

impl Dimensions {
    fn from_env() -> Self {
        fn dimension(name: &str, default: usize) -> usize {
            match std::env::var(name) {
                Ok(value) => value.parse().unwrap_or_else(|_| panic!("invalid {name}")),
                Err(std::env::VarError::NotPresent) => default,
                Err(error) => panic!("{name}: {error}"),
            }
        }
        let dims = Self {
            columns: dimension("PAGE_INDEX_BENCH_COLUMNS", 256),
            rows: dimension("PAGE_INDEX_BENCH_ROWS", 32768),
            row_group_rows: dimension("PAGE_INDEX_BENCH_ROW_GROUP_ROWS", 4096),
            page_rows: dimension("PAGE_INDEX_BENCH_PAGE_ROWS", 64),
            value_bytes: dimension("PAGE_INDEX_BENCH_VALUE_BYTES", 128),
        };
        assert!(
            dims.columns >= 6,
            "overlap requires at least six string columns"
        );
        assert!(dims.page_rows >= 8);
        assert!(dims.row_group_rows / dims.page_rows >= 4);
        assert_eq!(dims.row_group_rows % dims.page_rows, 0);
        assert!(dims.rows / dims.row_group_rows >= 2);
        assert_eq!(dims.rows % dims.row_group_rows, 0);
        assert!(i64::try_from(dims.rows).is_ok());
        assert!(dims.value_bytes >= 32);
        dims
    }

    fn json(&self) -> Value {
        json!({
            "columns": self.columns, "rows": self.rows,
            "row_group_rows": self.row_group_rows, "page_rows": self.page_rows,
            "value_bytes": self.value_bytes, "compression": "SNAPPY",
            "dictionary": false, "statistics": "Page", "fixture_version": 1
        })
    }

    fn schema(&self) -> SchemaRef {
        let mut fields = vec![
            Field::new("id", DataType::Int64, false),
            Field::new("residual", DataType::Int64, false),
        ];
        fields
            .extend((0..self.columns).map(|column| {
                Field::new(format!("c{column:03}"), DataType::Utf8, false)
            }));
        Arc::new(Schema::new(fields))
    }

    fn band(&self) -> (usize, usize) {
        let start = (self.rows / self.row_group_rows / 2) * self.row_group_rows
            + self.page_rows
            + self.page_rows / 4;
        (start, start + self.page_rows / 2 - 1)
    }
}

fn residual(id: usize) -> i64 {
    // Modulo predicate cannot use min/max page statistics; each page has a mix.
    ((id as u64 % 997 * 17 + 13) % 997) as i64
}

fn string_value(id: usize, column: usize, bytes: usize) -> String {
    // Long, varied bounds, not a short constant prefix that truncates away the
    // index payload. Disable bound truncation in the writer as well.
    let mut state = (id as u64).wrapping_mul(6364136223846793005)
        ^ (column as u64).wrapping_mul(1442695040888963407);
    let mut value = String::with_capacity(bytes);
    for _ in 0..bytes {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        value.push((b'a' + (state.wrapping_mul(2685821657736338717) % 26) as u8) as char);
    }
    value
}

fn generate_file(path: &Path, dims: &Dimensions) {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    let properties = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_column_index_truncate_length(Some(dims.value_bytes))
        .set_statistics_truncate_length(Some(dims.value_bytes))
        .set_max_row_group_row_count(Some(dims.row_group_rows))
        .set_data_page_row_count_limit(dims.page_rows)
        .set_write_batch_size(dims.page_rows.min(1024))
        .set_key_value_metadata(Some(vec![KeyValue::new(
            FIXTURE_KEY.to_owned(),
            dims.json().to_string(),
        )]))
        .build();
    let schema = dims.schema();
    let mut writer =
        ArrowWriter::try_new(file, schema.clone(), Some(properties)).unwrap();
    // Bound input batch memory independently of total fixture rows.
    let batch_rows = dims.page_rows.min(1024);
    for start in (0..dims.rows).step_by(batch_rows) {
        let end = (start + batch_rows).min(dims.rows);
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(
                (start..end).map(|id| id as i64),
            )),
            Arc::new(Int64Array::from_iter_values((start..end).map(residual))),
        ];
        arrays.extend((0..dims.columns).map(|column| {
            Arc::new(StringArray::from_iter_values(
                (start..end).map(|id| string_value(id, column, dims.value_bytes)),
            )) as ArrayRef
        }));
        writer
            .write(&RecordBatch::try_new(schema.clone(), arrays).unwrap())
            .unwrap();
    }
    writer.close().unwrap();
}

fn validate_file(path: &Path, dims: &Dimensions) -> Value {
    let metadata = ParquetMetaDataReader::new()
        .with_page_index_policy(PageIndexPolicy::Required)
        .parse_and_finish(&File::open(path).unwrap())
        .unwrap();
    let footer = metadata.file_metadata();
    let config = footer
        .key_value_metadata()
        .unwrap()
        .iter()
        .find(|entry| entry.key == FIXTURE_KEY)
        .and_then(|entry| entry.value.as_ref())
        .expect("not a page-index-cache benchmark fixture");
    assert_eq!(
        serde_json::from_str::<Value>(config).unwrap(),
        dims.json(),
        "existing fixture dimensions differ; use another file or matching environment"
    );
    assert_eq!(footer.num_rows() as usize, dims.rows);
    let schema =
        parquet_to_arrow_schema(footer.schema_descr(), footer.key_value_metadata())
            .unwrap();
    assert_eq!(schema.fields(), dims.schema().fields());
    assert_eq!(metadata.num_row_groups(), dims.rows / dims.row_group_rows);
    let indexes = metadata
        .page_index()
        .expect("fixture must have page indexes");
    let mut index_bytes = 0_i64;
    for (rg, group) in metadata.row_groups().iter().enumerate() {
        assert_eq!(group.num_rows() as usize, dims.row_group_rows);
        for (leaf, column) in group.columns().iter().enumerate() {
            assert_eq!(column.compression(), Compression::SNAPPY);
            assert!(column.dictionary_page_offset().is_none());
            let index = indexes.column_index(rg, leaf).unwrap();
            if leaf >= 2 {
                let ColumnIndexMetaData::BYTE_ARRAY(index) = index else {
                    panic!("string leaf must have byte-array page bounds");
                };
                for page in 0..dims.row_group_rows / dims.page_rows {
                    assert_eq!(index.min_value(page).unwrap().len(), dims.value_bytes);
                    assert_eq!(index.max_value(page).unwrap().len(), dims.value_bytes);
                }
            }
            let offsets = indexes.offset_index(rg, leaf).unwrap();
            assert_eq!(
                offsets.page_locations().len(),
                dims.row_group_rows / dims.page_rows
            );
            for (page, location) in offsets.page_locations().iter().enumerate() {
                assert_eq!(location.first_row_index as usize, page * dims.page_rows);
            }
            index_bytes += i64::from(column.column_index_length().unwrap());
            index_bytes += i64::from(column.offset_index_length().unwrap());
        }
    }
    assert!(index_bytes > 0);
    json!({"file_bytes": std::fs::metadata(path).unwrap().len(),
        "serialized_page_index_bytes": index_bytes,
        "all_column_index_entries": metadata.num_row_groups() * (dims.columns + 2),
        "all_offset_index_entries": metadata.num_row_groups() * (dims.columns + 2)})
}

struct Query {
    name: &'static str,
    pushdown: bool,
    band: bool,
    filter: bool,
}

impl Query {
    fn sql(&self, dims: &Dimensions, overlap: bool) -> String {
        let columns = if overlap { [0, 1, 4, 5] } else { [0, 1, 2, 3] };
        let projection = columns.map(|column| format!("c{column:03}")).join(", ");
        let (start, end) = dims.band();
        let predicate = match (self.band, self.filter) {
            (true, false) => format!("id BETWEEN {start} AND {end}"),
            (true, true) => format!("id BETWEEN {start} AND {end} AND residual % 7 = 0"),
            (false, true) => "residual % 7 = 0".to_owned(),
            (false, false) => unreachable!(),
        };
        format!("SELECT id, {projection} FROM t WHERE {predicate}")
    }

    fn validate(
        &self,
        batches: &[RecordBatch],
        dims: &Dimensions,
        overlap: bool,
    ) -> usize {
        let (start, end) = dims.band();
        let expected: Vec<_> = (0..dims.rows)
            .filter(|&id| {
                (!self.band || (start..=end).contains(&id))
                    && (!self.filter || residual(id) % 7 == 0)
            })
            .map(|id| id as i64)
            .collect();
        let columns = if overlap { [0, 1, 4, 5] } else { [0, 1, 2, 3] };
        let mut actual = Vec::with_capacity(expected.len());
        for batch in batches {
            assert_eq!(batch.num_columns(), 5);
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(ids.null_count(), 0);
            actual.extend_from_slice(ids.values());
            for (output, column) in columns.iter().enumerate() {
                let values = batch
                    .column(output + 1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                assert_eq!(values.null_count(), 0);
                for row in 0..batch.num_rows() {
                    assert_eq!(
                        values.value(row),
                        string_value(ids.value(row) as usize, *column, dims.value_bytes)
                    );
                }
            }
        }
        actual.sort_unstable();
        assert_eq!(
            actual, expected,
            "{} returned incorrect IDs/count",
            self.name
        );
        actual.len()
    }
}

async fn execute(
    ctx: &SessionContext,
    sql: &str,
) -> (Arc<dyn ExecutionPlan>, Vec<RecordBatch>) {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let batches = collect(plan.clone(), ctx.task_ctx()).await.unwrap();
    (plan, batches)
}

fn metrics(
    plan: &dyn ExecutionPlan,
    counts: &mut BTreeMap<String, usize>,
    times: &mut BTreeMap<String, usize>,
    pruning: &mut BTreeMap<String, [usize; 3]>,
) {
    if let Some(set) = plan.metrics() {
        // Iterate the raw MetricsSet: sum every partition/file label, rather than
        // taking the first counter or parsing the human-readable plan display.
        for metric in set.iter() {
            match metric.value() {
                MetricValue::Count { name, count } => {
                    *counts.entry(name.to_string()).or_default() += count.value();
                }
                MetricValue::Time { name, time } => {
                    *times.entry(name.to_string()).or_default() += time.value();
                }
                MetricValue::ElapsedCompute(time) => {
                    *times.entry("elapsed_compute".to_owned()).or_default() +=
                        time.value();
                }
                MetricValue::PruningMetrics {
                    name,
                    pruning_metrics,
                } => {
                    let value = pruning.entry(name.to_string()).or_default();
                    value[0] += pruning_metrics.pruned();
                    value[1] += pruning_metrics.matched();
                    value[2] += pruning_metrics.fully_matched();
                }
                _ => {}
            }
        }
    }
    for child in plan.children() {
        metrics(child.as_ref(), counts, times, pruning);
    }
}

fn cache_snapshot(ctx: &SessionContext) -> Value {
    let entries = ctx
        .runtime_env()
        .cache_manager
        .get_file_metadata_cache()
        .list_entries();
    let mut columns = 0;
    let mut offsets = 0;
    let mut bytes = 0;
    for entry in entries.values() {
        bytes += entry.size_bytes;
        let cached = entry
            .value
            .file_metadata
            .as_any()
            .downcast_ref::<CachedParquetMetaData>()
            .unwrap();
        let metadata = cached.parquet_metadata();
        if let Some(provider) = metadata.page_index() {
            for rg in 0..metadata.num_row_groups() {
                for leaf in 0..metadata.file_metadata().schema_descr().num_columns() {
                    columns += usize::from(provider.column_index(rg, leaf).is_some());
                    offsets += usize::from(provider.offset_index(rg, leaf).is_some());
                }
            }
        }
    }
    json!({"metadata_bytes": bytes, "files": entries.len(),
        "column_index_entries": columns, "offset_index_entries": offsets})
}

fn reset_and_seed_footer(rt: &Runtime, ctx: &SessionContext, path: &Path) {
    let cache = ctx.runtime_env().cache_manager.get_file_metadata_cache();
    cache.clear();
    let store = LocalFileSystem::new();
    let location = object_store::path::Path::from_filesystem_path(path).unwrap();
    rt.block_on(async {
        let meta = store.head(&location).await.unwrap();
        let footer = DFParquetMetadata::new(&store, &meta)
            .with_file_metadata_cache(Some(cache))
            .with_page_index_policy(Some(PageIndexPolicy::Skip))
            .fetch_metadata()
            .await
            .unwrap();
        assert!(footer.page_index().is_none());
    });
}

#[derive(Clone, Copy)]
enum CacheState {
    Cold,
    Identical,
    Overlap,
}

impl CacheState {
    fn label(self) -> &'static str {
        match self {
            Self::Cold => "cold_page_indexes",
            Self::Identical => "warm_identical",
            Self::Overlap => "warm_overlap",
        }
    }
}

fn prepare(
    rt: &Runtime,
    ctx: &SessionContext,
    path: &Path,
    seed_sql: &str,
    state: CacheState,
) {
    reset_and_seed_footer(rt, ctx, path);
    if !matches!(state, CacheState::Cold) {
        std::hint::black_box(rt.block_on(execute(ctx, seed_sql)));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let dims = Dimensions::from_env();
    let temp = tempfile::tempdir().unwrap();
    let path = std::env::var_os("PAGE_INDEX_BENCH_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| temp.path().join("wide.parquet"));
    if !path.exists() {
        generate_file(&path, &dims);
    }
    let path = path.canonicalize().unwrap();
    let fixture = validate_file(&path, &dims);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let queries = [
        Query {
            name: "prune_only",
            pushdown: false,
            band: true,
            filter: false,
        },
        Query {
            name: "prune_and_pushdown",
            pushdown: true,
            band: true,
            filter: true,
        },
        Query {
            name: "filter_only_control",
            pushdown: true,
            band: false,
            filter: true,
        },
    ];
    let mut observations = Vec::new();
    let mut group = c.benchmark_group(format!(
        "parquet_page_index_cache/{}cols_{}rows_rg{}_page{}_bytes{}",
        dims.columns, dims.rows, dims.row_group_rows, dims.page_rows, dims.value_bytes
    ));
    for query in queries {
        let mut config = SessionConfig::new().with_target_partitions(1);
        let parquet = &mut config.options_mut().execution.parquet;
        parquet.enable_page_index = true;
        parquet.pushdown_filters = query.pushdown;
        parquet.reorder_filters = false;
        parquet.schema_force_view_types = false;
        let ctx = SessionContext::new_with_config(config);
        // Avoid eviction of a full-width entry in either implementation.
        ctx.runtime_env()
            .cache_manager
            .get_file_metadata_cache()
            .update_cache_limit(1024 * 1024 * 1024);
        rt.block_on(ctx.register_parquet(
            "t",
            path.to_str().unwrap(),
            Default::default(),
        ))
        .unwrap();
        let seed_sql = query.sql(&dims, false);
        let (_, batches) = rt.block_on(execute(&ctx, &seed_sql));
        query.validate(&batches, &dims, false); // prime non-metadata caches as well
        for state in [CacheState::Cold, CacheState::Identical, CacheState::Overlap] {
            let overlap = matches!(state, CacheState::Overlap);
            let sql = query.sql(&dims, overlap);
            prepare(&rt, &ctx, &path, &seed_sql, state);
            let before = cache_snapshot(&ctx);
            let (plan, batches) = rt.block_on(execute(&ctx, &sql));
            let rows = query.validate(&batches, &dims, overlap);
            let after = cache_snapshot(&ctx);
            let mut counts = BTreeMap::new();
            let mut times = BTreeMap::new();
            let mut pruning = BTreeMap::new();
            metrics(plan.as_ref(), &mut counts, &mut times, &mut pruning);
            assert!(counts.get("bytes_scanned").copied().unwrap_or_default() > 0);
            if query.band {
                assert!(
                    pruning.get("page_index_rows_pruned").unwrap()[0] > 0,
                    "band must exercise page pruning, not just row-group pruning"
                );
            }
            if query.filter {
                assert!(
                    counts
                        .get("pushdown_rows_pruned")
                        .copied()
                        .unwrap_or_default()
                        > 0
                );
            }
            observations.push(json!({
                "query": query.name, "cache_state": state.label(), "sql": sql,
                "seed_sql": if matches!(state, CacheState::Cold) { None } else { Some(&seed_sql) },
                "pushdown_filters": query.pushdown, "query_rows": rows,
                "cache_before": before, "cache_after": after,
                "bytes_scanned": counts.get("bytes_scanned"),
                "pushdown_rows_pruned": counts.get("pushdown_rows_pruned"),
                "page_index_rows_pruned": pruning.get("page_index_rows_pruned").map(|value| value[0]),
                "count_metrics": counts, "time_metrics_ns": times,
                "pruning_metrics_pruned_matched_fully_matched": pruning
            }));
            // The diagnostics above are independent of Criterion's sampling.
            prepare(&rt, &ctx, &path, &seed_sql, state);
            group.bench_function(format!("{}/{}", query.name, state.label()), |b| {
                if matches!(state, CacheState::Identical) {
                    b.iter(|| std::hint::black_box(rt.block_on(execute(&ctx, &sql))));
                } else {
                    b.iter_batched(
                        || prepare(&rt, &ctx, &path, &seed_sql, state),
                        |()| std::hint::black_box(rt.block_on(execute(&ctx, &sql))),
                        BatchSize::PerIteration,
                    );
                }
            });
        }
    }
    group.finish();
    let report = json!({
        "dimensions": dims.json(), "fixture": fixture, "cases": observations,
        "target_partitions": 1, "metadata_cache_limit_bytes": 1024 * 1024 * 1024_u64,
        "timing": "SQL planning + execution + collect; setup and oracle excluded",
        "cold_scope": "page-index metadata only; OS, footer, listing and statistics warm"
    });
    let report = serde_json::to_string_pretty(&report).unwrap();
    println!("{report}");
    if let Some(report_path) = std::env::var_os("PAGE_INDEX_BENCH_REPORT") {
        let report_path = PathBuf::from(report_path);
        // A fresh destination also protects against symlink/hard-link aliases
        // of the fixture. Use a distinct report path for each invocation.
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(report_path)
            .expect("report path must not already exist");
        file.write_all(report.as_bytes()).unwrap();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
