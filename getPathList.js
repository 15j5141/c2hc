const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');
const { parse } = require('csv-parse');
console.log(parse);
const PATH_CODENET = './Project_CodeNet/';
const inputFiles = {}
const inputFilesPerQuestion = {} // 1問1答でサンプルコードを用意する.
class WaitFS {
    constructor() {
        this.isWait = false;
    }
    async waitFS() {
        this.isWait = true;
        while (this.isWait) {
            await this.wait(1);
        }
    }
    stop() {
        this.isWait = false;
    }
    wait(ms) {
        return new Promise((resolve, reject) => {
            setTimeout(resolve, ms)
        })
    }
}
const timer = new WaitFS();
const records = [];
const initParser = () => {
    const p = parse({ columns: true });
    p.on('error', function (err) {
        console.error(err.message);
    });
    p.on('readable', function () {
        let record;
        while ((record = p.read()) !== null) {
            // 何故かobjectになっているのでありがたくfilter処理する.
            if (record['language'] != 'C') continue;
            if (record['status'] != 'Accepted') continue;
            if (record['accuracy'] != '1/1') continue;
            // if (['s805099364'].includes(record['submission_id'])) continue;
            if (['p00017','p00104','p00121','p00685','p00692','p00697'].includes(record['problem_id'])) continue;

            // 最低1問1答あれば省略する. そうしないと10日位ビルドにかかる.
            if (inputFilesPerQuestion[record['problem_id']] != null) {
                continue;
            }
            inputFilesPerQuestion[record['problem_id']] = true;

            // 入力テキストがあれば読み込む.
            const inputTextPath = "derived/input_output/data/" + record['problem_id'] + "/input.txt"
            if (inputFiles[record['problem_id']] == null) {
                // ファイルが無ければ空文字を代入する.
                inputFiles[record['problem_id']] =
                    fsSync.existsSync(PATH_CODENET + inputTextPath)
                        ? fsSync.readFileSync(PATH_CODENET + inputTextPath, { encoding: "ascii" }).toString()
                        : "";
            }
            const r = {
                // "sid": record['submission_id'],
                // "pid": record['problem_id'],
                // "ext": record['filename_ext'],
                "c": "data/" + record['problem_id'] + "/C/" + record['submission_id'] + "." + record['filename_ext'],
                "in": inputFiles[record['problem_id']],
            }




            records.push(r);
        }
    });
    p.on('end', async () => {
        // console.log("endRead");
        timer.stop();
    });
    return p;
}

(async () => {
    const list = await fs.readdir('.');
    console.log(list);

    let parser = initParser();
    let fileNameNumber = "";
    for (let i = 0; i < 4053; i++) {
        fileNameNumber = "p" + String(i).padStart(5, '0') + ".csv";
        console.log("startRead:" + fileNameNumber);
        // ファイルが大きいのでstreamで処理する.
        fsSync.createReadStream(PATH_CODENET + 'metadata/' + fileNameNumber).pipe(parser);
        // 同期的にファイルを順番に読み込む.
        await timer.waitFS();
        // 再読み込みの為の処理を行う.
        parser.end();
        parser = initParser();
    }
    fsSync.writeFileSync("./records.json", JSON.stringify(records))
})();