# Docs for data collection for cliko-2 tests

`student_answer_fetch.sql` is the core SQL used to fetch response data from
2019/01/01 until (date run)

`student_answer_fetch.psql` (notice the 'p') is the same SQL, wrapped in a psql command:

`\copy (<query here>) TO '<filename>' with csv headers`

This takes advantage of postgresql built in conversion to csv.

Can probably run the later by piping it to `bin/rails/db -p`:

`cat student_answer_fetch.psql | bin/rails db -p`
