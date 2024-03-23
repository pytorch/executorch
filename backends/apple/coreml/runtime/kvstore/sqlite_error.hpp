//
// sqlite_error.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once


#include <types.hpp>

#include <sqlite3.h>

namespace executorchcoreml {
namespace sqlite {

/// The error codes representing sqlite error codes.
enum class ErrorCode: int32_t {
    ERROR = SQLITE_ERROR,
    INTERNAL = SQLITE_INTERNAL,
    PERM = SQLITE_PERM,
    ABORT = SQLITE_ABORT,
    BUSY = SQLITE_BUSY,
    LOCKED = SQLITE_LOCKED,
    NOMEM = SQLITE_NOMEM,
    READONLY = SQLITE_READONLY,
    INTERRUPT = SQLITE_INTERRUPT,
    IOERR = SQLITE_IOERR,
    CORRUPT = SQLITE_CORRUPT,
    NOTFOUND = SQLITE_NOTFOUND,
    FULL = SQLITE_FULL,
    CANTOPEN = SQLITE_CANTOPEN,
    PROTOCOL = SQLITE_PROTOCOL,
    EMPTY = SQLITE_EMPTY,
    SCHEMA = SQLITE_SCHEMA,
    TOOBIG = SQLITE_TOOBIG,
    CONSTRAINT = SQLITE_CONSTRAINT,
    MISMATCH = SQLITE_MISMATCH,
    MISUSE = SQLITE_MISUSE,
    NOLFS = SQLITE_NOLFS,
    AUTH = SQLITE_AUTH,
    FORMAT = SQLITE_FORMAT,
    RANGE = SQLITE_RANGE,
    NOTADB = SQLITE_NOTADB,
    NOTICE = SQLITE_NOTICE,
    WARNING = SQLITE_WARNING,
    ROW = SQLITE_ROW,
    DONE = SQLITE_DONE,
    
    // Extended
    ABORT_ROLLBACK = SQLITE_ABORT_ROLLBACK,
    BUSY_RECOVERY = SQLITE_BUSY_RECOVERY,
    BUSY_SNAPSHOT = SQLITE_BUSY_SNAPSHOT,
    CANTOPEN_CONVPATH = SQLITE_CANTOPEN_CONVPATH,
    CANTOPEN_FULLPATH = SQLITE_CANTOPEN_FULLPATH,
    CANTOPEN_ISDIR = SQLITE_CANTOPEN_ISDIR,
    CANTOPEN_NOTEMPDIR = SQLITE_CANTOPEN_NOTEMPDIR,
    CONSTRAINT_CHECK = SQLITE_CONSTRAINT_CHECK,
    CONSTRAINT_COMMITHOOK = SQLITE_CONSTRAINT_COMMITHOOK,
    CONSTRAINT_FOREIGNKEY = SQLITE_CONSTRAINT_FOREIGNKEY,
    CONSTRAINT_FUNCTION = SQLITE_CONSTRAINT_FUNCTION,
    CONSTRAINT_NOTNULL = SQLITE_CONSTRAINT_NOTNULL,
    CONSTRAINT_PRIMARYKEY = SQLITE_CONSTRAINT_PRIMARYKEY,
    CONSTRAINT_ROWID = SQLITE_CONSTRAINT_ROWID,
    CONSTRAINT_TRIGGER = SQLITE_CONSTRAINT_TRIGGER,
    CONSTRAINT_UNIQUE = SQLITE_CONSTRAINT_UNIQUE,
    CONSTRAINT_VTAB = SQLITE_CONSTRAINT_VTAB,
    CORRUPT_VTAB = SQLITE_CORRUPT_VTAB,
    IOERR_ACCESS = SQLITE_IOERR_ACCESS,
    IOERR_BLOCKED = SQLITE_IOERR_BLOCKED,
    IOERR_CHECKRESERVEDLOCK = SQLITE_IOERR_CHECKRESERVEDLOCK,
    IOERR_CLOSE = SQLITE_IOERR_CLOSE,
    IOERR_CONVPATH = SQLITE_IOERR_CONVPATH,
    IOERR_DELETE = SQLITE_IOERR_DELETE,
    IOERR_DELETE_NOENT = SQLITE_IOERR_DELETE_NOENT,
    IOERR_DIR_CLOSE = SQLITE_IOERR_DIR_CLOSE,
    IOERR_DIR_FSYNC = SQLITE_IOERR_DIR_FSYNC,
    IOERR_FSTAT = SQLITE_IOERR_FSTAT,
    IOERR_FSYNC = SQLITE_IOERR_FSYNC,
    IOERR_GETTEMPPATH = SQLITE_IOERR_GETTEMPPATH,
    IOERR_LOCK = SQLITE_IOERR_LOCK,
    IOERR_MMAP = SQLITE_IOERR_MMAP,
    IOERR_NOMEM = SQLITE_IOERR_NOMEM,
    IOERR_RDLOCK = SQLITE_IOERR_RDLOCK,
    IOERR_READ = SQLITE_IOERR_READ,
    IOERR_SEEK = SQLITE_IOERR_SEEK,
    IOERR_SHMLOCK = SQLITE_IOERR_SHMLOCK,
    IOERR_SHMMAP = SQLITE_IOERR_SHMMAP,
    IOERR_SHMOPEN = SQLITE_IOERR_SHMOPEN,
    IOERR_SHMSIZE = SQLITE_IOERR_SHMSIZE,
    IOERR_SHORT_READ = SQLITE_IOERR_SHORT_READ,
    IOERR_TRUNCATE = SQLITE_IOERR_TRUNCATE,
    IOERR_UNLOCK = SQLITE_IOERR_UNLOCK,
    IOERR_WRITE = SQLITE_IOERR_WRITE,
    LOCKED_SHAREDCACHE = SQLITE_LOCKED_SHAREDCACHE,
    NOTICE_RECOVER_ROLLBACK = SQLITE_NOTICE_RECOVER_ROLLBACK,
    NOTICE_RECOVER_WAL = SQLITE_NOTICE_RECOVER_WAL,
    OK_LOAD_PERMANENTLY = SQLITE_OK_LOAD_PERMANENTLY,
    READONLY_CANTLOCK = SQLITE_READONLY_CANTLOCK,
    READONLY_DBMOVED = SQLITE_READONLY_DBMOVED,
    READONLY_RECOVERY = SQLITE_READONLY_RECOVERY,
    READONLY_ROLLBACK = SQLITE_READONLY_ROLLBACK,
    WARNING_AUTOINDEX = SQLITE_WARNING_AUTOINDEX
};

enum class Status {
    success = 1,
    error,
    internal,
    perm,
    abort,
    busy,
    locked,
    nomem,
    readonly,
    interrupt,
    ioerr,
    corrupt,
    notfound,
    full,
    cantopen,
    protocol,
    empty,
    schema,
    toobig,
    constraint,
    mismatch,
    misuse,
    nolfs,
    auth,
    format,
    range,
    notadb,
    notice,
    warning,
    row,
    done
};

/// The category for a sqlite error.
struct ErrorCategory final : std::error_category {
    inline const char* name() const noexcept override {
        return "sqlite";
    }
    
    inline std::string message(int v) const override {
        return std::string(sqlite3_errstr(v));
    }
    
    static inline ErrorCategory& instance() {
        static ErrorCategory c;
        return c;
    }
    
    bool equivalent(const std::error_code& ec, int condition) const noexcept override;
};


inline std::error_code make_error_code(ErrorCode code) {
    return {static_cast<int>(code), ErrorCategory::instance()};
}

/// Checks if the sqlite status is not an error.
///
/// @param status The sqlite status.
/// @retval `true` if the status is not an error otherwise `false`.
bool is_status_ok(int status);

/// Processes sqlite return status.
///
/// @param status The sqlite status.
/// @param error   if the sqlite status is an error then it is converted to `error_code` and is assigned to `error`.
/// @retval `true` if the status is not an error otherwise `false`.
bool process_sqlite_status(int status, std::error_code& error);

} // namespace sqlite
} // namespace executorchcoreml

namespace std {

template <>
struct is_error_code_enum<executorchcoreml::sqlite::ErrorCode> : true_type {};

}  // namespace std
