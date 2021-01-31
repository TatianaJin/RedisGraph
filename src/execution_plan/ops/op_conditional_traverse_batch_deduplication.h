/*
* Copyright 2021 HDL
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#pragma once

#include <stdio.h>

#include "../../../deps/GraphBLAS/Include/GraphBLAS.h"
#include "../../arithmetic/algebraic_expression.h"
#include "../execution_plan.h"
#include "op.h"
#include "shared/traverse_functions.h"

/* OP Traverse with intra-batch deduplication*/
typedef struct {
  OpBase op;
  Graph *graph;
  AlgebraicExpression *ae;
  GrB_Matrix F;               // Filter matrix.
  GrB_Matrix M;               // Algebraic expression result.
  NodeID dest_label_id;       // ID of destination node label if known.
  const char *dest_label;     // Label of destination node if known.
  EdgeTraverseCtx *edge_ctx;  // Edge collection data if the edge needs to be set.
  GxB_MatrixTupleIter *iter;  // Iterator over M.
  int srcNodeIdx;             // Source node index into record.
  int destNodeIdx;            // Destination node index into record.
  uint record_count;          // Number of held records.
  uint record_cap;            // Max number of records to process.
  Record *records;            // Array of records.
  Record r;                   // Currently selected record.

  FILE *stats_file;     // file handler for recording traversal counts
  uint *record_offset;  // size (record_cap + 1), result row index to record offset
  uint group_idx;       // the index of the current record in the current group
  uint group_end;       // the index of the last record in the current group
  uint edge_size;       // current edge index for edge population
  Node dest_node;       // current destination node
} OpCondTraverseBD;

/* Creates a new Traverse operation */
OpBase *NewCondTraverseOpBD(const ExecutionPlan *plan, Graph *g, AlgebraicExpression *ae);
