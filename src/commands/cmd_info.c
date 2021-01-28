/*
* Copyright 2018-2020 HDL
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "cmd_info.h"

#include "../config.h"
#include "../errors.h"
#include "../query_ctx.h"
#include "cmd_context.h"

void GrB_Matrix_GetNonEmptyRows(size_t *nrows, const GrB_Matrix A, const GrB_Vector non_empty_rows) {
  assert(nrows != NULL);
  assert(A != NULL);
  assert(non_empty_rows != NULL);
  assert(GrB_reduce(non_empty_rows, NULL, NULL, GxB_ANY_BOOL_MONOID, A, NULL) == GrB_SUCCESS);
  assert(GrB_reduce(nrows, NULL, GxB_PLUS_UINT64_MONOID, non_empty_rows, NULL) == GrB_SUCCESS);
}

// only approximation
size_t GrB_Matrix_GetMemory(GrB_Matrix m, uint element_size) {
  size_t n;
  size_t memory = 10 * sizeof(int64_t) + sizeof(GrB_Type);  // metadata
  assert(GrB_Matrix_nrows(&n, m) == GrB_SUCCESS);
  memory += (n + 1) * sizeof(int64_t) * 2;  // vlist
  assert(GrB_Matrix_nvals(&n, m) == GrB_SUCCESS);
  memory += n * sizeof(int64_t) + n * element_size;  // elist and values
  return memory;
}

// Return memory usage of datablock in bytes
size_t DataBlock_GetMemoryUsage(const DataBlock *datablock) {
  size_t total_memory = 0;
  total_memory += sizeof(uint64_t) * 2;                                                 // itemCount and itemCap
  total_memory += sizeof(uint) * 2;                                                     // itemSize and blockCount
  total_memory += sizeof(pthread_mutex_t);                                              // mutex
  total_memory += sizeof(fpDestructor);                                                 // destructor
  total_memory += sizeof(uint64_t *) + array_sizeof(array_hdr(datablock->deletedIdx));  // deletedIdx
  total_memory += sizeof(Block **) + sizeof(Block *) * datablock->blockCount +
                  (DATABLOCK_BLOCK_CAP * datablock->itemSize + sizeof(Block)) * datablock->blockCount;  // blocks
  // TODO(tatiana): include memory pointed to by property SIValue
  return total_memory;
}

// Return the length of the entire output string, including chars before offset
int DataBlock_GetSizes(const DataBlock *datablock, char *output_str, int output_limit, int offset) {
  offset +=
      snprintf(output_str + offset, output_limit - offset, "%ld items (%dB each) %ld deleted %ld capacity\n",
               datablock->itemCount, datablock->itemSize, (size_t)array_len(datablock->deletedIdx), datablock->itemCap);
  offset +=
      snprintf(output_str + offset, output_limit - offset, "%d blocks (%dB each) total %ldB", datablock->blockCount,
               datablock->itemSize * DATABLOCK_BLOCK_CAP, DataBlock_GetMemoryUsage(datablock));
  return offset;
}

void _ReportCountInfo(const Graph *graph, RedisModuleCtx *ctx, GraphContext *gc) {
  size_t n_nodes, n_edges;
  int n_labels, n_edge_types;
  size_t n_labeled_nodes, n_typed_edges;
  size_t n_nodes_with_typed_out_edge;
  size_t n_nodes_with_typed_in_edge;
  GrB_Vector ones;
  bool one = true;
  size_t nrows;
  char str[1024];

  RedisModule_ReplyWithArray(ctx, 4);

  // Get overall counts
  n_nodes = Graph_NodeCount(graph);
  n_edges = Graph_EdgeCount(graph);
  n_labels = Graph_LabelTypeCount(graph);
  n_edge_types = Graph_RelationTypeCount(graph);
  int len = snprintf(str, 1024, "V=%ld, E=%ld, L=%d, T=%d", n_nodes, n_edges, n_labels, n_edge_types);
  RedisModule_ReplyWithStringBuffer(ctx, str, len);

  // Get node/edge counts by type
  RedisModule_ReplyWithArray(ctx, n_labels);  // an nested array
  for (int i = 0; i < n_labels; ++i) {
    n_labeled_nodes = Graph_LabeledNodeCount(graph, i);
    int len = snprintf(str, 1024, ":%s %ld nodes", gc->node_schemas[i]->name, n_labeled_nodes);
    RedisModule_ReplyWithStringBuffer(ctx, str, len);
  }
  RedisModule_ReplyWithArray(ctx, n_edge_types);  // an nested array
  for (int i = 0; i < n_edge_types; ++i) {
    GrB_Matrix m = Graph_GetRelationMatrix(graph, i);
    assert(m);
    GrB_Matrix_nvals(&n_typed_edges, m);
    int len = snprintf(str, 1024, ":%s %ld edges", gc->relation_schemas[i]->name, n_typed_edges);
    RedisModule_ReplyWithStringBuffer(ctx, str, len);
  }

  // Get node counts by edge types
  bool maintain_transpose;
  Config_Option_get(Config_MAINTAIN_TRANSPOSE, &maintain_transpose);
  assert(GrB_Vector_new(&ones, GrB_BOOL, n_nodes) == GrB_SUCCESS);
  RedisModule_ReplyWithArray(ctx, n_edge_types);  // an nested array
  if (maintain_transpose) {
    for (int i = 0; i < n_edge_types; ++i) {
      GrB_Matrix m = Graph_GetRelationMatrix(graph, i);
      GrB_Matrix mt = Graph_GetTransposedRelationMatrix(graph, i);
      assert(m);
      assert(mt);
      GrB_Matrix_nrows(&nrows, m);
      assert(nrows == n_nodes);
      GrB_Matrix_GetNonEmptyRows(&n_nodes_with_typed_out_edge, m, ones);
      GrB_Matrix_GetNonEmptyRows(&n_nodes_with_typed_in_edge, mt, ones);
      int len = snprintf(str, 1024, ":%s %ld src %ld dest", gc->relation_schemas[i]->name, n_nodes_with_typed_out_edge,
                         n_nodes_with_typed_in_edge);
      RedisModule_ReplyWithStringBuffer(ctx, str, len);
    }
  } else {
    for (int i = 0; i < n_edge_types; ++i) {
      GrB_Matrix m = Graph_GetRelationMatrix(graph, i);
      assert(m);
      GrB_Matrix_nrows(&nrows, m);
      assert(nrows == n_nodes);
      GrB_Matrix_GetNonEmptyRows(&n_nodes_with_typed_out_edge, m, ones);
      int len = snprintf(str, 1024, ":%s %ld src", gc->relation_schemas[i]->name, n_nodes_with_typed_out_edge);
      RedisModule_ReplyWithStringBuffer(ctx, str, len);
    }
  }
  GrB_free(&ones);
}

void _ReportMemoryInfo(const Graph *graph, RedisModuleCtx *ctx) {
  char str[1024];
  int len = 0;
  size_t total_memory = 0;
  RedisModule_ReplyWithArray(ctx, 5);

  bool maintain_transpose;
  Config_Option_get(Config_MAINTAIN_TRANSPOSE, &maintain_transpose);

  len = snprintf(str, 1024, "nodes\n");
  len = DataBlock_GetSizes(graph->nodes, str, 1024, len);
  RedisModule_ReplyWithStringBuffer(ctx, str, len);
  total_memory += DataBlock_GetMemoryUsage(graph->nodes);

  len = snprintf(str, 1024, "edges\n");
  len = DataBlock_GetSizes(graph->edges, str, 1024, len);
  RedisModule_ReplyWithStringBuffer(ctx, str, len);

  len = snprintf(str, 1024, "adjacency_matrix ");
  size_t mem = GrB_Matrix_GetMemory(Graph_GetAdjacencyMatrix(graph), sizeof(bool));
  total_memory += mem;
  len += snprintf(str + len, 1024 - len, "%ld", mem);
  RedisModule_ReplyWithStringBuffer(ctx, str, len);

  if (maintain_transpose) {
    len = snprintf(str, 1024, "adjacency_matrix_t ");
    size_t mem = GrB_Matrix_GetMemory(Graph_GetTransposedAdjacencyMatrix(graph), sizeof(bool));
    total_memory += mem;
    len += snprintf(str + len, 1024 - len, "%ld", mem);
    RedisModule_ReplyWithStringBuffer(ctx, str, len);
  }

  int n_labels = Graph_LabelTypeCount(graph);
  for (int i = 0; i < n_labels; ++i) {
    total_memory += GrB_Matrix_GetMemory(Graph_GetLabelMatrix(graph, i), sizeof(bool));
  }
  int n_edge_types = Graph_RelationTypeCount(graph);
  for (int i = 0; i < n_labels; ++i) {
    total_memory += GrB_Matrix_GetMemory(Graph_GetRelationMatrix(graph, i), sizeof(uint64_t));
  }
  if (maintain_transpose) {
    for (int i = 0; i < n_labels; ++i) {
      total_memory += GrB_Matrix_GetMemory(Graph_GetTransposedRelationMatrix(graph, i), sizeof(uint64_t));
    }
  }
  size_t B, KB, MB, GB;
  B = total_memory % 1024;
  total_memory /= 1024;
  KB = total_memory % 1024;
  total_memory /= 1024;
  MB = total_memory % 1024;
  total_memory /= 1024;
  GB = total_memory;
  len = snprintf(str, 1024, "total memory excluding properties %ldG%ldM%ldK%ldB", GB, MB, KB, B);
  RedisModule_ReplyWithStringBuffer(ctx, str, len);
}

/** Returns the graph information including stats and memory consumption.
 * Readonly operation.
 */
void Graph_Info(void *args) {
  CommandCtx *command_ctx = (CommandCtx *)args;
  RedisModuleCtx *ctx = CommandCtx_GetRedisCtx(command_ctx);
  GraphContext *gc = CommandCtx_GetGraphContext(command_ctx);

  CommandCtx_TrackCtx(command_ctx);

  // Acquire read lock for it is an readonly operation
  Graph_AcquireReadLock(gc->g);

  // Set policy after lock acquisition, avoid resetting policies between readers and writers.
  Graph_SetMatrixPolicy(gc->g, SYNC_AND_MINIMIZE_SPACE);

  const Graph *graph = gc->g;
  RedisModule_ReplyWithArray(ctx, 2);
  _ReportCountInfo(graph, ctx, gc);
  _ReportMemoryInfo(graph, ctx);

  // cleanup
  Graph_ReleaseLock(gc->g);

  GraphContext_Release(gc);
  CommandCtx_Free(command_ctx);
}
