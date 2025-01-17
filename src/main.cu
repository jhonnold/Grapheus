
#include "chess/chess.h"
#include "dataset/batchloader.h"
#include "dataset/dataset.h"
#include "dataset/io.h"
#include "dataset/process.h"
#include "misc/csv.h"
#include "misc/timer.h"
#include "nn/nn.h"
#include "operations/operations.h"

#include <fstream>

using namespace nn;
using namespace data;

struct ChessModel : nn::Model {

    // seting inputs
    virtual void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions) = 0;

    // train function
    void train(dataset::BatchLoader<chess::Position>& loader,
               int                                    epochs     = 1000,
               int                                    epoch_size = 1e7) {
        this->compile(loader.batch_size);

        Timer t {};
        for (int i = 0; i < epochs; i++) {
            t.start();
            size_t prev_duration = 0;
            float  batch_loss    = 0;
            float  epoch_loss    = 0;

            for (int b = 0; b < epoch_size / loader.batch_size; b++) {
                // get the next dataset and set it up while the other things
                // are running on the gpu
                auto* ds = loader.next();
                setup_inputs_and_outputs(ds);

                // print score of last iteration
                if (b > 0) {
                    batch_loss = loss_of_last_batch();
                    epoch_loss += batch_loss;
                }

                t.stop();
                if (b > 0
                    && (b == (epoch_size / loader.batch_size) - 1
                        || t.elapsed() - prev_duration > 1000)) {
                    prev_duration = t.elapsed();

                    printf("\rep/ba = [%3d/%5d], ", i, b);
                    printf("batch_loss = [%1.8f], ", batch_loss);
                    printf("epoch_loss = [%1.8f], ", epoch_loss / b);
                    printf("speed = [%9d pos/s], ",
                           (int) round(1000.0f * loader.batch_size * b / t.elapsed()));
                    printf("time = [%3ds]", (int) t.elapsed() / 1000);
                    std::cout << std::flush;
                }

                // start training of new batch
                batch();
            }

            std::cout << std::endl;
            next_epoch(epoch_loss / (epoch_size / loader.batch_size));
        }
    }

    void test_fen(const std::string& fen){
        this->compile(1);

        chess::Position pos = chess::parse_fen(fen);
        dataset::DataSet<chess::Position> ds{};
        ds.positions.push_back(pos);
        ds.header.entry_count = 1;

        // setup inputs of network
        setup_inputs_and_outputs(&ds);

        // forward pass
        this->upload_inputs();
        this->forward();

        // go through the layers and download values

        std::cout << "==================================================================================\n";
        std::cout << "testing fen: " << fen << std::endl;

        int idx = 0;
        for(auto layer:m_layers){
            layer->dense_output.values >> CPU;

            std::cout << "LAYER " << ++idx << std::endl;
            for(int i = 0; i < std::min((size_t)16, layer->size); i++){
                std::cout << std::setw(10) << layer->dense_output.values(i, 0);
            }
            if(layer->size > 16){
                std::cout << " ......... " << layer->dense_output.values(layer->size - 1, 0);
            }
            std::cout << "\n";
        }
    }

    void distribution(dataset::BatchLoader<chess::Position>& loader, int batches = 32){
        this->compile(loader.batch_size);

        std::vector<DenseMatrix<float>> max_values{};
        std::vector<DenseMatrix<float>> min_values{};

        for(auto l:m_layers){
            max_values.emplace_back(l->dense_output.values.m, 1);
            min_values.emplace_back(l->dense_output.values.m, 1);
            max_values.back().malloc<data::CPU>();
            min_values.back().malloc<data::CPU>();
            math::uniform(max_values.back(), -1000000.0f, -1000000.0f);
            math::uniform(min_values.back(),  1000000.0f,  1000000.0f);
        }

        for (int b = 0; b < batches; b++) {
            auto* ds = loader.next();
            setup_inputs_and_outputs(ds);
            this->upload_inputs();
            this->forward();
            std::cout << "\r" << b << " / " << batches << std::flush;

            // get minimum and maximum values
            for(int i = 0; i < m_layers.size(); i++) {
                auto layer = m_layers[i].get();
                layer->dense_output.values >> data::CPU;
                for(int m =0; m < layer->dense_output.values.m; m++){
                    for(int n =0; n < layer->dense_output.values.n; n++){
                        max_values[i](m,0) = std::max(max_values[i](m,0),  layer->dense_output.values(m,n));
                        min_values[i](m,0) = std::min(min_values[i](m,0),  layer->dense_output.values(m,n));
                    }
                }
            }
        }
        std::cout << std::endl;


        for(int i = 0; i < m_layers.size(); i++) {
            std::cout << "------------ LAYER " << i+1 << " --------------------" << std::endl;
            std::cout << "min: ";
            for(int j = 0; j < std::min((size_t)16, min_values[i].size()); j++){
                std::cout << std::setw(10) << min_values[i](j);
            }
            if(min_values[i].size() > 16){
                std::cout << " ......... " << min_values[i](min_values.size()-1);
            }
            std::cout << "\n";

            std::cout << "max: ";
            for(int j = 0; j < std::min((size_t)16, max_values[i].size()); j++){
                std::cout << std::setw(10) << max_values[i](j);
            }
            if(max_values[i].size() > 16){
                std::cout << " ......... " << max_values[i](max_values.size()-1);
            }

            std::cout << "\n";
            float min =  10000000;
            float max = -10000000;
            for(int m =0; m < min_values.size(); m++){
                min = std::min(min, min_values[i](m));
                max = std::max(max, max_values[i](m));
            }
            std::cout << "output bounds: [" << min << " ; " << max << "]\n";


            int died = 0;
            for(int j = 0; j < max_values[i].size(); j++){
                if(std::abs(max_values[i](j) - min_values[i](j)) < 1e-8){
                    died ++;
                }
            }

            std::cout << "died: " << died << " / " << max_values[i].size();
            std::cout << "\n";

            for(auto p : m_layers[i]->params()){
                float min =  10000000;
                float max = -10000000;
                for(int m =0; m < p->values.m; m++){
                    for(int n =0; n < p->values.n; n++){
                        min = std::min(min, p->values(m,n));
                        max = std::max(max, p->values(m,n));
                    }
                }

                std::cout << "param bounds: [" << min << " ; " << max << "]\n";
            }

        }

    }
};

struct KoiModel : ChessModel {
    static constexpr int THREADS = 16;    // threads to use on the cpu

    SparseInput*         in1;
    SparseInput*         in2;

    KoiModel() : ChessModel() {
        in1     = add<SparseInput>(16 * 12 * 64, 32);
        in2     = add<SparseInput>(16 * 12 * 64, 32);

        auto ft = add<FeatureTransformer>(in1, in2, 512);
        auto re = add<ReLU>(ft);
        auto af = add<Affine>(re, 1);
        auto sm = add<Sigmoid>(af, 2.5 / 400);

        set_loss(MPE {2.5, false});
        set_lr_schedule(StepDecayLRSchedule {0.01, 0.3, 100});
        add_optimizer(Adam({{OptimizerEntry {&ft->weights}},
                            {OptimizerEntry {&ft->bias}},
                            {OptimizerEntry {&af->weights}},
                            {OptimizerEntry {&af->bias}}},
                           0.9,
                           0.999,
                           1e-8));

        set_file_output("../res/test/");
        add_quantization(Quantizer {
            "quant_1",
            10,
            QuantizerEntry<int16_t>(&ft->weights.values, 32, true),
            QuantizerEntry<int16_t>(&ft->bias.values   , 32),
            QuantizerEntry<int16_t>(&af->weights.values, 128),
            QuantizerEntry<int32_t>(&af->bias.values   , 128 * 32),
        });
        set_save_frequency(10);
    }

    inline int king_square_index(chess::Square relative_king_square) {

        // clang-format off
        constexpr int indices[chess::N_SQUARES] {
            0,  1,  2,  3,  3,  2,  1,  0,
            4,  5,  6,  7,  7,  6,  5,  4,
            8,  9,  10, 11, 11, 10, 9,  8,
            8,  9,  10, 11, 11, 10, 9,  8,
            12, 12, 13, 13, 13, 13, 12, 12,
            12, 12, 13, 13, 13, 13, 12, 12,
            14, 14, 15, 15, 15, 15, 14, 14,
            14, 14, 15, 15, 15, 15, 14, 14,
        };
        // clang-format on

        return indices[relative_king_square];
    }

    inline int index(chess::Square piece_square,
                     chess::Piece  piece,
                     chess::Square king_square,
                     chess::Color  view) {
        constexpr int          PIECE_TYPE_FACTOR  = 64;
        constexpr int          PIECE_COLOR_FACTOR = PIECE_TYPE_FACTOR * 6;
        constexpr int          KING_SQUARE_FACTOR = PIECE_COLOR_FACTOR * 2;

        const chess::PieceType piece_type         = chess::type_of(piece);
        const chess::Color     piece_color        = chess::color_of(piece);

        chess::Square          relative_king_square;
        chess::Square          relative_piece_square;

        if (view == chess::WHITE) {
            relative_king_square  = king_square;
            relative_piece_square = piece_square;
        } else {
            relative_king_square  = chess::mirror_ver(king_square);
            relative_piece_square = chess::mirror_ver(piece_square);
        }

        const int king_square_idx = king_square_index(relative_king_square);
        if (chess::file_index(king_square) > 3) {
            relative_piece_square = chess::mirror_hor(relative_piece_square);
        }

        const int index = relative_piece_square + piece_type * PIECE_TYPE_FACTOR
                          + (piece_color == view) * PIECE_COLOR_FACTOR
                          + king_square_idx * KING_SQUARE_FACTOR;
        return index;
    }

    void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto& target = m_loss->target;

#pragma omp parallel for schedule(static) num_threads(16)
        for (int b = 0; b < positions->header.entry_count; b++) {
            chess::Position* pos = &positions->positions[b];
            // fill in the inputs and target values

            chess::Square wKingSq = pos->get_king_square<chess::WHITE>();
            chess::Square bKingSq = pos->get_king_square<chess::BLACK>();

            chess::BB     bb {pos->m_occupancy};
            int           idx = 0;

            while (bb) {
                chess::Square sq                    = chess::lsb(bb);
                chess::Piece  pc                    = pos->m_pieces.get_piece(idx);

                auto          piece_index_white_pov = index(sq, pc, wKingSq, chess::WHITE);
                auto          piece_index_black_pov = index(sq, pc, bKingSq, chess::BLACK);

                if (pos->m_meta.stm() == chess::WHITE) {
                    in1->sparse_output.set(b, piece_index_white_pov);
                    in2->sparse_output.set(b, piece_index_black_pov);
                } else {
                    in2->sparse_output.set(b, piece_index_white_pov);
                    in1->sparse_output.set(b, piece_index_black_pov);
                }

                bb = chess::lsb_reset(bb);
                idx++;
            }

            float p_value = pos->m_result.score;
            float w_value = pos->m_result.wdl;

            // flip if black is to move -> relative network style
            if (pos->m_meta.stm() == chess::BLACK) {
                p_value = -p_value;
                w_value = -w_value;
            }

            float p_target = 1 / (1 + expf(-p_value * 2.5 / 400));
            float w_target = (w_value + 1) / 2.0f;

            target(b)      = (p_target + w_target) / 2.0f;
        }
    }
};

int main() {
    init();

    DenseInput inp{8};
    ChunkwiseMul m{&inp, 4};

    inp.compile(8);
    m  .compile(8);

    math::uniform(inp.dense_output.values, 0.f, 1.f);

    inp.dense_output.values >> GPU;
    inp.forward();
    m.forward();

    m.dense_output.values >> CPU;

    std::cout << inp.dense_output.values << std::endl;
    std::cout << m  .dense_output.values << std::endl;

//    init();
//    std::vector<std::string> files {};
//
//    for (int i = 1; i <= 32; i++) {
//        files.push_back(R"(D:\Koivisto Resourcen\Training Data Shuffled + Berserk\koi_ber_)"
//                        + std::to_string(i) + ".bin");
//    }
//    dataset::BatchLoader<chess::Position> loader {files, 16384};
//    loader.start();
//
//    KoiModel model {};
//    model.load_weights("../res/run4/weights/680.state");
////    model.quantize("680_32_128.net");
////    model.load_weights(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\CudAD\resources\runs\experiment_37\weights-epoch1000.nnue)");
////    model.train(loader, 1000, 1e8);
//    model.distribution(loader, 32);

//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//    model.compile(16384);
//    model.setup_inputs_and_outputs(loader.next());
//    model.batch();
//    std::cout << model.loss_of_last_batch() << std::endl;



//    model.load_weights("../res/run1/weights/test.state");
//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//    model.train(loader, 1000, 1e8);
//    model.load_weights("../res/run2/weights/1000.state");
//    model.distribution(loader);
//    model.quantize("final_16_256.net");

//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//    model.save_weights("../res/run1/weights/test.state");

//    model.compile(1);
//    model.load_weights(R"(C:\Users\Luecx\CLionProjects\Grapheus\res\run1\weights\300.state)");
//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//
//    model.load_weights(R"(C:\Users\Luecx\CLionProjects\Grapheus\res\run1\weights\200.state)");
//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");


//    model.train(loader, 1000, 1e8);
//    loader.kill();
//
    close();
    return 0;
}
