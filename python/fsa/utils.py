import numpy as np
from elftools.elf.enums import *
from elftools.elf.constants import P_FLAGS
from elftools.elf.structs import ELFStructs


def compare_matrices(ref: tuple[str, np.ndarray], impls: dict[str, np.ndarray]):
    """比较多个实现与参考实现的误差"""

    def error_metrics(a, b):
        return {
            "MAE": np.mean(np.abs(a - b)),  # 平均绝对误差
            "MSE": np.mean((a - b) ** 2),  # 均方误差
            "MaxErr": np.max(np.abs(a - b)),  # 最大误差
            "RelErr": np.mean(np.abs((a - b) / (b + 1e-8))),  # 相对误差
        }

    ref_name, ref_data = ref
    for name, data in impls.items():
        err = error_metrics(data, ref_data)
        print(f"Error of {name} vs {ref_name}:", err)


class DictToClass:
    """将字典转换为类属性,方便pyelftools库使用"""

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                # 递归转换嵌套字典
                v = DictToClass(v)
            self.__setattr__(k, v)


"""
ELF文件布局:
    ┌─────────────────┐ 0x00
    │   ELF Header    │ (64字节)
    ├─────────────────┤ 0x40
    │ Program Header  │ (56字节 x N段)
    │     Table       │
    ├─────────────────┤
    │  .shstrtab      │ 节名字符串表
    │   (节数据)      │
    ├─────────────────┤ ← data_offset
    │   Segment 1     │ 段数据
    │   Segment 2     │
    │      ...        │
    ├─────────────────┤ ← section_header_offset
    │ Section Header  │ (64字节 x 2节)
    │     Table       │
    └─────────────────┘
"""


class ElfWriter:
    def __init__(self, segments: list[tuple[int, int, bytes]], alignment: int):
        """
        segments: list[(BaseAddr, Size, Bytes)]
            BaseAddr: 内存中的虚拟地址(RISC-V地址空间)
            Size: 段大小
            Bytes: 段数据
        alignment: 内存对齐要求, 通常为页大小(如4KB)
        """
        self.structs = ELFStructs(elfclass=64)  # 64位ELF
        self.structs.create_basic_structs()
        self.structs.create_advanced_structs(
            e_type=ENUM_E_TYPE["ET_NONE"],  # 未指定类型
            e_machine=ENUM_E_MACHINE["EM_RISCV"],  # RISC-V架构
            e_ident_osabi=0,  # 无特定OS ABI
        )
        """
        ELF layout:
        0x00: ELF header
        0x40: program header table
        0x40 + 56 * len(segments): section data (shstrtab content)
        0x40 + 56 * len(segments) + len(shstrtab): segment data
        segments[-1].p_offset + segments[-1].p_filesz: section header table

        """
        # 格式:\x00 + "dummy_section" + \x00 + ".shstrtab" + \x00
        # 这是ELF文件中的字符串表,存储节名称
        self.shstrtab_data = b"\x00dummy_section\x00.shstrtab\x00"
        self.dummy_section_name_offset = 1  # Points to "dummy_section"
        self.shstrtab_name_offset = 15  # Points to ".shstrtab"
        self.data_offset = 64 + 56 * len(segments) + len(self.shstrtab_data)
        self.data_alignment = alignment
        self.segments: list[dict] = [
            self.__add_segment(addr, size, data) for (addr, size, data) in segments
        ]
        if len(self.segments) > 0:
            self.section_header_offset = (
                self.segments[-1]["p_offset"] + self.segments[-1]["p_filesz"]
            )
        else:
            self.section_header_offset = self.data_offset

    def __align(self, offset: int) -> int:
        if offset % self.data_alignment != 0:
            return offset + (self.data_alignment - (offset % self.data_alignment))
        return offset

    def __add_segment(self, addr: int, size: int, data: bytes) -> dict:
        self.data_offset = self.__align(self.data_offset)
        segment = {
            "p_type": ENUM_P_TYPE_RISCV["PT_LOAD"],  # 可加载段
            "p_offset": self.data_offset,  # 在文件中的偏移
            "p_vaddr": addr,  # 虚拟地址
            "p_paddr": addr,  # 物理地址
            "p_filesz": size,  # 文件中的大小
            "p_memsz": size,  # 内存中的大小
            "p_flags": P_FLAGS.PF_R | P_FLAGS.PF_W,  # 可读可写
            "p_align": self.data_alignment,  # 对齐要求
            "data": data,  # 实际数据
        }
        self.data_offset += size
        return segment

    def write_elf(self, filename: str):
        with open(filename, "wb") as f:
            # ELF header
            f.write(
                self.structs.Elf_Ehdr.build(
                    DictToClass(
                        {
                            "e_ident": {
                                "EI_MAG": b"\x7fELF",
                                "EI_CLASS": "ELFCLASS64",
                                "EI_DATA": "ELFDATA2LSB",
                                "EI_VERSION": 1,
                                "EI_OSABI": 0,
                                "EI_ABIVERSION": 0,
                                "EI_PAD": bytes(7),
                            },
                            "e_type": ENUM_E_TYPE["ET_EXEC"],
                            "e_machine": ENUM_E_MACHINE["EM_RISCV"],
                            "e_version": ENUM_E_VERSION["EV_CURRENT"],
                            "e_entry": 0,
                            "e_phoff": 64 if self.segments else 0,
                            "e_shoff": self.section_header_offset,
                            "e_flags": 0,
                            "e_ehsize": 64,
                            "e_phentsize": 56,
                            "e_phnum": len(self.segments),
                            "e_shentsize": 64,
                            "e_shnum": 2,
                            "e_shstrndx": 1,
                        }
                    )
                )
            )
            # program header table
            for seg in self.segments:
                f.write(self.structs.Elf_Phdr.build(DictToClass(seg)))
            # section data (.shstrtab content)
            f.write(self.shstrtab_data)
            # segment data
            for seg in self.segments:
                cur_offset = f.tell()
                padding = seg["p_offset"] - cur_offset
                if padding > 0:
                    f.write(b"\x00" * padding)
                f.write(seg["data"])
            assert f.tell() == self.section_header_offset
            # section header table
            # dummy_section
            f.write(
                self.structs.Elf_Shdr.build(
                    DictToClass(
                        {
                            "sh_name": self.dummy_section_name_offset,
                            "sh_type": ENUM_SH_TYPE_RISCV["SHT_NULL"],
                            "sh_flags": 0,
                            "sh_addr": 0,
                            "sh_offset": 0,
                            "sh_size": 0,
                            "sh_link": 0,
                            "sh_info": 0,
                            "sh_addralign": 0,
                            "sh_entsize": 0,
                        }
                    )
                )
            )
            # .shstrtab
            f.write(
                self.structs.Elf_Shdr.build(
                    DictToClass(
                        {
                            "sh_name": self.shstrtab_name_offset,
                            "sh_type": ENUM_SH_TYPE_RISCV["SHT_STRTAB"],
                            "sh_flags": 0,
                            "sh_addr": 0,
                            "sh_offset": 64 + 56 * len(self.segments),
                            "sh_size": len(self.shstrtab_data),
                            "sh_link": 0,
                            "sh_info": 0,
                            "sh_addralign": 1,
                            "sh_entsize": 0,
                        }
                    )
                )
            )
