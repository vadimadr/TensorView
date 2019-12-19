#pragma once

#include <string>
#include <sstream>


#define ASSERT_1(expr) do { if(!!(expr)) ; else assertion_error(#expr, __FILE__, __LINE__ ); } while(0);
#define ASSERT_2(expr, msg) do { if(!!(expr)) ; else assertion_error(#expr, msg, __FILE__, __LINE__ ); } while(0);

#ifndef NDEBUG

#define ASSERT_DEBUG_1(expr) ASSERT_1(expr)
#define ASSERT_DEBUG_2(expr, msg) ASSERT_2(expr, msg)

#else

#define ASSERT_DEBUG_1(expr)
#define ASSERT_DEBUG_2(expr, msg)


#endif

// Dispatch implementation
#define GET_MACRO(_1, _2, NAME, ...) NAME

#define TV_ASSERT_DEBUG(...) GET_MACRO(__VA_ARGS__, ASSERT_DEBUG_2, ASSERT_DEBUG_1) (__VA_ARGS__)
#define TV_ASSERT(...) GET_MACRO(__VA_ARGS__, ASSERT_2, ASSERT_1)(__VA_ARGS__)

static void assertion_error(const std::string& expr, const std::string& fname, int line) {
    std::stringstream ss;
    ss << fname << ':' << line << ": Assertion \"" << expr << "\" failed";
    throw std::runtime_error(ss.str());
}

static void assertion_error(const std::string& expr, const std::string& msg, const std::string& fname, int line) {
    std::stringstream ss;
    ss << fname << ':' << line << ": Assertion \"" << expr << "\" failed: " << msg;
    throw std::runtime_error(ss.str());
}

