// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

TEST_CASE("SparsityPattern::createFaceToMatrixAddress", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

    DYNAMIC_SECTION("" << size)
    {
        BENCHMARK(std::string(execName) + "_construct")
        {
            return NeoN::la::createSparsityPatternFaceToMatrixAddress<NeoN::localIdx>(mesh);
        };
    }
}
