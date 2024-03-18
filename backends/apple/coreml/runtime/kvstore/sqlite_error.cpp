//
// sqlite_error.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.
//

#include <sqlite_error.hpp>

#include <sqlite3.h>

namespace executorchcoreml {
namespace sqlite {

bool ErrorCategory::equivalent(const std::error_code& code, int condition) const noexcept {
    switch (static_cast<Status>(condition)) {
        case Status::success: {
            return !code || (code == ErrorCode::OK_LOAD_PERMANENTLY) || (code == ErrorCode::ROW) ||
                (code == ErrorCode::DONE);
        }
        case Status::error: {
            return (code == ErrorCode::ERROR);
        }
        case Status::internal: {
            return (code == ErrorCode::INTERNAL);
        }
        case Status::perm: {
            return (code == ErrorCode::PERM);
        }
        case Status::abort: {
            return (code == ErrorCode::ABORT) || (code == ErrorCode::ABORT_ROLLBACK);
        }
        case Status::busy: {
            return (code == ErrorCode::BUSY) || (code == ErrorCode::BUSY_RECOVERY) ||
                (code == ErrorCode::BUSY_SNAPSHOT);
        }
        case Status::locked: {
            return (code == ErrorCode::LOCKED) || (code == ErrorCode::LOCKED_SHAREDCACHE);
        }
        case Status::nomem: {
            return (code == ErrorCode::NOMEM);
        }
        case Status::readonly: {
            return (code == ErrorCode::READONLY) || (code == ErrorCode::READONLY_CANTLOCK) ||
                (code == ErrorCode::READONLY_DBMOVED) || (code == ErrorCode::READONLY_RECOVERY) ||
                (code == ErrorCode::READONLY_ROLLBACK);
        }
        case Status::interrupt: {
            return (code == ErrorCode::INTERRUPT);
        }
        case Status::ioerr: {
            return (code == ErrorCode::IOERR) || (code == ErrorCode::IOERR_ACCESS) ||
                (code == ErrorCode::IOERR_BLOCKED) || (code == ErrorCode::IOERR_CHECKRESERVEDLOCK) ||
                (code == ErrorCode::IOERR_CLOSE) || (code == ErrorCode::IOERR_CONVPATH) ||
                (code == ErrorCode::IOERR_DELETE) || (code == ErrorCode::IOERR_DELETE_NOENT) ||
                (code == ErrorCode::IOERR_DIR_CLOSE) || (code == ErrorCode::IOERR_DIR_FSYNC) ||
                (code == ErrorCode::IOERR_FSTAT) || (code == ErrorCode::IOERR_FSYNC) ||
                (code == ErrorCode::IOERR_GETTEMPPATH) || (code == ErrorCode::IOERR_LOCK) ||
                (code == ErrorCode::IOERR_MMAP) || (code == ErrorCode::IOERR_NOMEM) ||
                (code == ErrorCode::IOERR_RDLOCK) || (code == ErrorCode::IOERR_READ) ||
                (code == ErrorCode::IOERR_SEEK) || (code == ErrorCode::IOERR_SHMLOCK) ||
                (code == ErrorCode::IOERR_SHMMAP) || (code == ErrorCode::IOERR_SHMOPEN) ||
                (code == ErrorCode::IOERR_SHMSIZE) || (code == ErrorCode::IOERR_SHORT_READ) ||
                (code == ErrorCode::IOERR_TRUNCATE) || (code == ErrorCode::IOERR_UNLOCK) ||
                (code == ErrorCode::IOERR_WRITE);
        }
        case Status::corrupt: {
            return (code == ErrorCode::CORRUPT) || (code == ErrorCode::CORRUPT_VTAB);
        }
        case Status::notfound: {
            return (code == ErrorCode::NOTFOUND);
        }
        case Status::full: {
            return (code == ErrorCode::FULL);
        }
        case Status::cantopen: {
            return (code == ErrorCode::CANTOPEN) || (code == ErrorCode::CANTOPEN_CONVPATH) ||
                (code == ErrorCode::CANTOPEN_FULLPATH) || (code == ErrorCode::CANTOPEN_ISDIR) ||
                (code == ErrorCode::CANTOPEN_NOTEMPDIR);
        }
        case Status::protocol: {
            return (code == ErrorCode::PROTOCOL);
        }
        case Status::empty: {
            return (code == ErrorCode::EMPTY);
        }
        case Status::schema: {
            return (code == ErrorCode::SCHEMA);
        }
        case Status::toobig: {
            return (code == ErrorCode::TOOBIG);
        }
        case Status::constraint: {
            return (code == ErrorCode::CONSTRAINT) || (code == ErrorCode::CONSTRAINT_CHECK) ||
                (code == ErrorCode::CONSTRAINT_COMMITHOOK) || (code == ErrorCode::CONSTRAINT_FOREIGNKEY) ||
                (code == ErrorCode::CONSTRAINT_FUNCTION) || (code == ErrorCode::CONSTRAINT_NOTNULL) ||
                (code == ErrorCode::CONSTRAINT_PRIMARYKEY) || (code == ErrorCode::CONSTRAINT_ROWID) ||
                (code == ErrorCode::CONSTRAINT_TRIGGER) || (code == ErrorCode::CONSTRAINT_UNIQUE) ||
                (code == ErrorCode::CONSTRAINT_VTAB);
        }
        case Status::mismatch: {
            return (code == ErrorCode::MISMATCH);
        }
        case Status::misuse: {
            return (code == ErrorCode::MISUSE);
        }
        case Status::nolfs: {
            return (code == ErrorCode::NOLFS);
        }
        case Status::auth: {
            return (code == ErrorCode::AUTH);
        }
        case Status::format: {
            return (code == ErrorCode::FORMAT);
        }
        case Status::range: {
            return (code == ErrorCode::RANGE);
        }
        case Status::notadb: {
            return (code == ErrorCode::NOTADB);
        }
        case Status::notice: {
            return (code == ErrorCode::NOTICE) || (code == ErrorCode::NOTICE_RECOVER_ROLLBACK) ||
                (code == ErrorCode::NOTICE_RECOVER_WAL);
        }
        case Status::warning: {
            return (code == ErrorCode::WARNING) || (code == ErrorCode::WARNING_AUTOINDEX);
        }
        case Status::row: {
            return (code == ErrorCode::ROW);
        }
        case Status::done: {
            return (code == ErrorCode::DONE);
        }
        default: {
            return false;
        }
    }
}

bool is_status_ok(int status) {
    return (status == SQLITE_OK || status == SQLITE_DONE || status == SQLITE_OK_LOAD_PERMANENTLY ||
            status == SQLITE_ROW);
}

bool process_sqlite_status(int status, std::error_code& code) {
    if (is_status_ok(status)) {
        return true;
    }
    code = std::error_code(status, ErrorCategory::instance());
    return false;
}

} // namespace sqlite
} // namespace executorchcoreml
