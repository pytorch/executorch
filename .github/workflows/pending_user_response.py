import datetime
import os

from github import Github

REPO_NAME = "pytorch/executorch"
LABEL = "need-user-input"
REMINDER_MARKER = "<!-- executorch-auto-reminder -->"

DAYS_BEFORE_REMINDER = 30
DAYS_BEFORE_CLOSE = 30
REMINDER_COOLDOWN_DAYS = 7
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"

REMINDER_COMMENT = (
    REMINDER_MARKER
    + f"\nHi @{{0}}, this issue/PR has been marked as 'need-user-input'. "
    + f"Please respond or provide input. If we don't hear back in {DAYS_BEFORE_REMINDER} days, this will be closed."
)
CLOSE_COMMENT = (
    REMINDER_MARKER
    + f"\nClosing due to no response after {DAYS_BEFORE_CLOSE} days. "
    + "If you still need help, feel free to re-open or comment again!"
)


def main():
    g = Github(os.environ["GH_TOKEN"])
    repo = g.get_repo(REPO_NAME)

    print(f"[DRY_RUN={DRY_RUN}] Fetching open issues with label '{LABEL}'.")
    issues = repo.get_issues(state="open", labels=[LABEL])

    now = datetime.datetime.now(datetime.timezone.utc)

    for issue in issues:
        print(f"Processing issue/PR #{issue.number}: {issue.title}")

        comments = sorted(
            issue.get_comments(),
            key=lambda comment: comment.created_at,
        )

        auto_comments = [
            comment for comment in comments if REMINDER_MARKER in (comment.body or "")
        ]
        user_comments = [
            comment for comment in comments if REMINDER_MARKER not in (comment.body or "")
        ]

        # ---- REMINDER LOGIC ----
        recent_auto_reminder = any(
            (now - comment.created_at).days < REMINDER_COOLDOWN_DAYS
            for comment in auto_comments
        )

        last_comment = comments[-1] if comments else None

        if not auto_comments:
            if (
                last_comment
                and (now - last_comment.created_at).days >= DAYS_BEFORE_REMINDER
            ):
                user = issue.user.login
                message = REMINDER_COMMENT.format(user)
                print(f"Posting first reminder for {user}")
                if not DRY_RUN:
                    issue.create_comment(message)
        elif not recent_auto_reminder:
            last_auto = auto_comments[-1]
            user = issue.user.login
            if (now - last_auto.created_at).days >= REMINDER_COOLDOWN_DAYS:
                message = REMINDER_COMMENT.format(user)
                print(f"Posting follow-up reminder for {user}")
                if not DRY_RUN:
                    issue.create_comment(message)

        # ---- CLOSE / LABEL LOGIC ----
        if auto_comments:
            first_auto_comment = auto_comments[0]
            days_since_first = (now - first_auto_comment.created_at).days

            user_responded = any(
                comment.created_at > first_auto_comment.created_at
                and comment.user.login == issue.user.login
                for comment in user_comments
            )

            if user_responded:
                print(f"User responded; remove '{LABEL}' label")
                if not DRY_RUN:
                    issue.remove_from_labels(LABEL)
            elif days_since_first >= DAYS_BEFORE_CLOSE:
                print(
                    f"Close issue/PR due to inactivity "
                    f"({days_since_first} days since first reminder)"
                )
                if not DRY_RUN:
                    issue.create_comment(CLOSE_COMMENT)
                    issue.edit(state="closed")
                    issue.remove_from_labels(LABEL)


if __name__ == "__main__":
    main()
