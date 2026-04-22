// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

TEMPLATE_TEST_CASE("LinearSystem", "[bench]", NeoN::scalar, NeoN::Vec3)

{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

    DYNAMIC_SECTION("" << size)
    {
        SECTION("Construction")
        {
            BENCHMARK(std::string(execName) + "_construct")
            {
                auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);
                return ls.matrix().values().size();
            };
        }

        SECTION("Reset")

        {
            auto ls = la::createEmptyLinearSystem<TestType>(mesh);
            BENCHMARK(std::string(execName) + "_reset") { ls.reset(); };
        }
    }
}
