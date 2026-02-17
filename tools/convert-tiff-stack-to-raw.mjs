import fs from 'node:fs';
import path from 'node:path';
import * as UTIFModule from 'utif2';

const UTIF = UTIFModule.default ?? UTIFModule;

function usage() {
    console.error('Usage: node tools/convert-tiff-stack-to-raw.mjs <input-dir> [output-base-path-without-extension]');
    process.exit(1);
}

function isTiff(name) {
    return /\.(tif|tiff)$/i.test(name);
}

function tagVal(ifd, key, fallback) {
    const v = ifd[key];
    if (v == null) return fallback;
    if (typeof v === 'number') return v;
    if (Array.isArray(v)) return Number(v[0]) || fallback;
    return fallback;
}

async function main() {
    const inputDir = process.argv[2];
    if (!inputDir) usage();
    const absInputDir = path.resolve(inputDir);
    if (!fs.existsSync(absInputDir) || !fs.statSync(absInputDir).isDirectory()) {
        throw new Error(`Input directory not found: ${absInputDir}`);
    }

    const files = fs.readdirSync(absInputDir)
        .filter(isTiff)
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));

    if (files.length === 0) {
        throw new Error(`No .tif/.tiff files found in ${absInputDir}`);
    }

    const outputBaseArg = process.argv[3];
    const outputBase = outputBaseArg
        ? path.resolve(outputBaseArg)
        : path.join(absInputDir, path.basename(absInputDir));
    const outputRaw = `${outputBase}.raw`;
    const outputMeta = `${outputBase}.raw.volumeinfo`;

    const firstBuffer = fs.readFileSync(path.join(absInputDir, files[0]));
    const firstIfds = UTIF.decode(firstBuffer.buffer.slice(firstBuffer.byteOffset, firstBuffer.byteOffset + firstBuffer.byteLength));
    if (!firstIfds.length) throw new Error(`No images found in ${files[0]}`);
    const firstIfd = firstIfds[0];
    UTIF.decodeImage(firstBuffer.buffer.slice(firstBuffer.byteOffset, firstBuffer.byteOffset + firstBuffer.byteLength), firstIfd);

    const width = firstIfd.width;
    const height = firstIfd.height;
    const bps = tagVal(firstIfd, 't258', 8);
    const spp = tagVal(firstIfd, 't277', 1);
    const photo = tagVal(firstIfd, 't262', 1);
    const isGray16 = spp === 1 && (photo === 0 || photo === 1) && bps === 16;
    const isGray8 = spp === 1 && (photo === 0 || photo === 1) && bps === 8;

    if (!isGray16 && !isGray8) {
        throw new Error(`Unsupported TIFF format in first slice: bps=${bps}, spp=${spp}, photo=${photo}`);
    }

    const bytesPerVoxel = isGray16 ? 2 : 1;
    const dataType = isGray16 ? 'uint16' : 'uint8';
    const expectedSliceBytes = width * height * bytesPerVoxel;

    console.log(`Input: ${absInputDir}`);
    console.log(`Slices: ${files.length}`);
    console.log(`Slice dimensions: ${width} x ${height}`);
    console.log(`Data type: ${dataType}`);
    console.log(`Output RAW: ${outputRaw}`);
    console.log(`Output metadata: ${outputMeta}`);

    const out = fs.createWriteStream(outputRaw, { flags: 'w' });

    for (let i = 0; i < files.length; i++) {
        const filePath = path.join(absInputDir, files[i]);
        const buf = fs.readFileSync(filePath);
        const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        const ifds = UTIF.decode(ab);
        if (!ifds.length) throw new Error(`No image found in ${files[i]}`);
        const ifd = ifds[0];
        UTIF.decodeImage(ab, ifd);
        if (ifd.width !== width || ifd.height !== height) {
            throw new Error(`Dimension mismatch in ${files[i]}: ${ifd.width}x${ifd.height}, expected ${width}x${height}`);
        }

        let sliceBytes;
        if (isGray16) {
            const data = new Uint16Array(ifd.data.buffer, ifd.data.byteOffset, width * height);
            sliceBytes = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
        } else {
            const rgba = UTIF.toRGBA8(ifd);
            const gray = new Uint8Array(width * height);
            for (let p = 0, g = 0; p < rgba.length; p += 4, g++) {
                gray[g] = rgba[p];
            }
            sliceBytes = Buffer.from(gray.buffer, gray.byteOffset, gray.byteLength);
        }

        if (sliceBytes.byteLength !== expectedSliceBytes) {
            throw new Error(`Decoded byte size mismatch in ${files[i]}: got ${sliceBytes.byteLength}, expected ${expectedSliceBytes}`);
        }

        if (!out.write(sliceBytes)) {
            await new Promise((resolve) => out.once('drain', resolve));
        }

        if ((i + 1) % 50 === 0 || i + 1 === files.length) {
            const pct = ((i + 1) / files.length * 100).toFixed(1);
            console.log(`Processed ${i + 1}/${files.length} (${pct}%)`);
        }
    }

    await new Promise((resolve, reject) => {
        out.end(() => resolve());
        out.on('error', reject);
    });

    const metaText = [
        '[Volume]',
        `SizeX=${width}`,
        `SizeY=${height}`,
        `SizeZ=${files.length}`,
        `Datatype=${dataType}`,
        'VoxelSizeX=1',
        'VoxelSizeY=1',
        'VoxelSizeZ=1',
        '',
    ].join('\n');
    fs.writeFileSync(outputMeta, metaText, 'utf8');

    const rawStat = fs.statSync(outputRaw);
    console.log(`Done. RAW size: ${rawStat.size} bytes`);
}

main().catch((err) => {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
});
